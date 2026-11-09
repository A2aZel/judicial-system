# Judicial AI — structured analysis backend

Версионируемый движок на FastAPI: формальные проверки черновика наказания и детерминированный поиск аналогов по структурированным признакам `analysis.csv`. Qwen и генеративные ответы не используются. Lazy encoder — `deepvk/RuModernBERT-base`; appeal risk по умолчанию выключен.

**Это не production-ready судебная рекомендательная система.** Извлечение CSV автоматическое и не проверено вручную в полном объёме. Формальные проверки не заменяют юриста, не проверяют санкции по редакции закона и не устанавливают причинность. Аналоги — описательная статистика небольшой выборки, а не прогноз решения суда.

## Локальный запуск

Требуется Python 3.12. Из корня репозитория:

```bash
python -m pip install -r requirements-dev.txt
python app.py
```

Для Linux x86_64 воспроизводимая установка со всеми транзитивными зависимостями и hashes:

```bash
python -m pip install --require-hashes -r requirements-dev-lock.txt
```

- Swagger: <http://localhost:8000/docs>
- OpenAPI: <http://localhost:8000/openapi.json>
- Health: <http://localhost:8000/api/v1/health>
- UI: <http://localhost:8000/> (`ui/index.html`, если файл существует)
- Статика UI: `/static/*`

По умолчанию читается `corpus/sudact/analysis.csv`. Сырой корпус не копируется в тесты или Docker-образ; `fixtures/synthetic_analysis.csv` полностью синтетический.

## Данные

Общее описание корпуса — `corpus/README.md`. Московские данные собираются из официальных
открытых наборов, а не из ограниченной robots-поисковой выдачи:

```bash
python tools/mosgorsud_open_data.py run --limit 10 --pause 4
```

Команда возобновляет сбор с checkpoint. Источники и правила — `corpus/mosgorsud/README.md`,
текущие агрегаты — `corpus/mosgorsud/report.md`. Каталог `corpus/mosgorsud/raw/` содержит
реквизиты дел и персональные данные, исключён из Git и не должен передаваться наружу.

## Docker / Compose

```bash
cp .env.example .env
docker compose up --build
```

Compose по умолчанию публикует порт только на `127.0.0.1`. Host `analysis.csv` монтируется read-only в `/data/analysis.csv`, кэш Hugging Face — в именованный volume `model_cache`. Путь к CSV меняется через `ANALYSIS_CSV_PATH`, порт — через `APP_PORT`. Контейнер работает под UID/GID `10001`, имеет read-only корневую ФС, `no-new-privileges` и healthcheck.

Для сетевого развёртывания необходимо одновременно задать все три значения в `.env` и поставить TLS/reverse proxy:

```dotenv
APP_BIND_ADDRESS=0.0.0.0
JUDICIAL_AI_APP_ENV=production
JUDICIAL_AI_API_KEY=replace-with-a-secret-from-your-secret-manager
```

Без непустого `JUDICIAL_AI_API_KEY` production-конфигурация не запускается. Не используйте пример выше как реальный секрет.

Сборка без Compose:

```bash
docker build -t judicial-ai:0.1.0 .
docker run --rm -p 127.0.0.1:8000:8000 \
  -v "$PWD/corpus/sudact/analysis.csv:/data/analysis.csv:ro" \
  -v judicial_model_cache:/models judicial-ai:0.1.0
```

### Воспроизводимость образа

Обе стадии используют `python:3.12.10-slim` с реальным linux/amd64 manifest digest `sha256:97983fa8cc88343512862c62307159a82261c3528dc025f79e5a3f7af43e50b4`, полученным через `docker buildx imagetools inspect python:3.12.10-slim`. Образ намеренно фиксирует linux/amd64.

`requirements-lock.txt` содержит 50 runtime-пакетов, `requirements-dev-lock.txt` — 53 dev-пакета, включая транзитивные зависимости и полученные из registry SHA256. `torch==2.13.0+cpu` разрешён из `https://download.pytorch.org/whl/cpu`; CUDA/NVIDIA wheels не входят в lock. Docker и CI используют `--require-hashes`.

Перегенерация lock после осознанного обновления прямых зависимостей:

```bash
uv pip compile requirements.txt --python-platform x86_64-manylinux_2_28 \
  --python-version 3.12 --torch-backend cpu --generate-hashes \
  --emit-index-url --emit-index-annotation --annotation-style line \
  --output-file requirements-lock.txt
uv pip compile requirements-dev.txt --python-platform x86_64-manylinux_2_28 \
  --python-version 3.12 --torch-backend cpu --generate-hashes \
  --emit-index-url --emit-index-annotation --annotation-style line \
  --output-file requirements-dev-lock.txt
```

После генерации сохраните `--extra-index-url https://download.pytorch.org/whl/cpu` в обоих lock-файлах: `uv` указывает CPU index в provenance torch, но не всегда эмитит его как глобальную опцию.

## Конфигурация

Все настройки имеют префикс `JUDICIAL_AI_`; примеры — в `.env.example`.

| Переменная | По умолчанию | Назначение |
|---|---|---|
| `APP_ENV` | `development` | `development`, `test`, `production` |
| `API_KEY` | пусто | Bearer key для `/models` и `/analyses`; обязателен в `production` |
| `CORPUS_PATH` | `corpus/sudact/analysis.csv` | Локальный CSV |
| `MAX_REQUEST_BYTES` | `65536` | Суммарный потоковый лимит HTTP request body |
| `DEFAULT_TOP_K` | `5` | Размер возвращаемого списка аналогов |
| `BERT_MODEL_ID` | `deepvk/RuModernBERT-base` | Закреплённый репозиторий encoder |
| `BERT_REVISION` | `8aee6f818db9070b635a549580dc64dcfa03e39a` | Полный immutable commit SHA |
| `BERT_RISK_ENABLED` | `false` | Явное разрешение scoring |
| `BERT_RISK_CHECKPOINT` | пусто | `risk_head.npz` или каталог с NPZ и JSON |
| `BERT_RISK_CHECKPOINT_SHA256` | пусто | SHA256, одобренный deployment config |
| `BERT_CACHE_DIR` | пусто | Кэш моделей; `/models` в Docker |
| `LOG_LEVEL` | `INFO` | Уровень логов |

Пустой API key отключает авторизацию только вне production. Health остаётся публичным. Bearer-сравнение выполняется над UTF-8 bytes, поэтому non-ASCII токен возвращает 401, а не 500. Логи содержат UUID анализа, счётчики и длительность, но не `narrative`, тексты дел, УИД или персональные данные.

## API

### `POST /api/v1/analyses`

```json
{
  "case": {
    "article_part": "3",
    "amount_rub": 150000,
    "episodes": 1,
    "special_order": false,
    "pretrial_agreement": false,
    "art_64": false,
    "art_69": false,
    "mitigating": [],
    "aggravating": [],
    "narrative": null
  },
  "draft_punishment": {
    "prison_months": 24,
    "fine_rub": null,
    "fine_multiple": null,
    "ban_years": 2,
    "conditional": false,
    "confiscation": false
  },
  "top_k": 5
}
```

Ответ содержит `analysis_id`, `legal_checks`, `analogues`, `punishment_summary`, `appeal_risk`, `warnings`, `model_info`. В аналоге есть только `distance` и исходы наказания — нет ID документа, УИД, ФИО, названия суда или текста. `analysis_id` — новый UUID запроса, не идентификатор дела.

`distance` может быть `null`, если внутри cohort нет активных различающих признаков. В этом случае ranking отключён, а список является лишь неранжированной подвыборкой размером до `top_k`.

Все числовые сводки включают `count`, `median`, `q1`, `q3`, `iqr`, `min`, `max` и **всегда рассчитываются по всей отфильтрованной cohort**, а не по top-k. Контракт обозначен `punishment_summary.statistics_scope="cohort"`; `candidate_count` — размер cohort, `returned_count` — размер списка. Поэтому статистика не зависит от `top_k` и порядка строк CSV. Отсутствие значений не заменяется нулями. При пустой cohort возвращаются пустые списки и нулевые счётчики с предупреждением.

Если задан API key:

```bash
curl -H "Authorization: Bearer $JUDICIAL_AI_API_KEY" \
  -H "Content-Type: application/json" \
  -d @request.json http://localhost:8000/api/v1/analyses
```

### Другие endpoints

- `GET /api/v1/health` — `ok` или `degraded`, число загруженных записей; всегда HTTP 200 для orchestration healthcheck.
- `GET /api/v1/models` — версии движка и ML-библиотек, backbone/revision, отдельные состояния загрузки/inference, kNN и риска.
- `GET /docs`, `GET /openapi.json` — интерактивная схема.

ASGI middleware ограничивает суммарное число байт по мере чтения, включая chunked запросы без `Content-Length`, и не вызывает неограниченный `request.body()`.

## kNN и ingestion

Внутренний `CaseRecord` не хранит `id`, `uid`, суд, период, ФИО или сырой текст. Остаются только `convicted_290=True` и записи с распознанной частью ст. 290. Точные hard filters: часть статьи, `special_order`, `pretrial_agreement`, `art_64`, `art_69`.

Обязательные булевы режимы принимают только известные true/false-представления. Пустое или неизвестное значение не превращается в `False`: запись исключается со счётчиком `skipped_records`. Числа должны быть конечными и находиться в диапазонах API; `NaN`, `inf`, отрицательные значения, дробные/выходящие за диапазон counters и `OverflowError` отклоняются.

Числовые признаки активируются только при наличии CSV-колонок: `amount_rub`, `episodes`, `mitigating_count`, `aggravating_count`. Сумма логарифмируется; признаки масштабируются по медиане/IQR с fallback на стандартное отклонение. Пропуски заполняются медианой cohort с warning. При наличии активных признаков используется евклидово расстояние и детерминированная сортировка.

В текущем `analysis.csv` нет этих числовых колонок. Булевы режимы уже константны после hard filters, поэтому ranking отключается: `distance=null`, warning, неранжированный список и cohort-статистика. Движок не придумывает суммы или эпизоды. Поле `conditional` — автоматический proxy из `art_73`, требующий ручной проверки. Отсутствие срока/штрафа в CSV может быть следствием извлечения или обезличивания, а не отсутствия наказания.

## Правовой модуль

Проверяются только полнота основного наказания, формальная связь условного режима со сроком, неоднозначная одновременная запись штрафа в рублях и кратной величине, а также предупреждение при `art_69=true` и одном эпизоде. Пределы санкций, действующая редакция закона, квалификация и обязательные обстоятельства имеют статус `not_checked`.

## Encoder и appeal risk

Default encoder и полная ревизия закреплены в Settings, Compose и `.env.example`. `AutoTokenizer`/`AutoModel` загружаются lazy; базовые веса разрешены для representations/embeddings, но не для вероятности риска.

Lead сообщил отдельный реальный CPU smoke для этой ревизии: Python 3.12.10, `torch 2.13.0+cpu`, output shape `[1, 11, 768]`, все значения finite; при `sliding_window=128` effective `local_attention=256`. Это **проверенная конкретная среда, а не универсальная гарантия** для других transformers, платформ, входов или конфигураций. Текущий unit-test suite не скачивает backbone.

### Deployment gates для risk

Для `appeal_risk.score` одновременно требуются:

1. `BERT_RISK_ENABLED=true`.
2. Полный 40-символьный commit SHA backbone; `main`, `None` и mutable tags отклоняются.
3. Отдельно обученный ненулевой logistic head `risk_head.npz` с `weight` и `bias`, finite values, без pickle.
4. SHA256 файла, одобренный через `BERT_RISK_CHECKPOINT_SHA256` deployment config. Фактический digest вычисляется потоково и сравнивается до загрузки backbone.
5. Совпадающий `weights_sha256` в `risk_head.json`, model/revision/task/architecture.
6. Зафиксированный preprocessing: tokenizer model/revision, CLS pooling, `add_special_tokens=true`, `normalization=none`, `overflow=reject`, `max_tokens`.

Пример metadata (digest должен быть реальным):

```json
{
  "weights_sha256": "actual-64-hex-sha256",
  "architecture_version": 1,
  "task": "binary_appeal_risk",
  "base_model_id": "deepvk/RuModernBERT-base",
  "base_revision": "8aee6f818db9070b635a549580dc64dcfa03e39a",
  "preprocessing": {
    "tokenizer_model_id": "deepvk/RuModernBERT-base",
    "tokenizer_revision": "8aee6f818db9070b635a549580dc64dcfa03e39a",
    "pooling": "cls",
    "max_tokens": 512,
    "overflow": "reject",
    "normalization": "none",
    "add_special_tokens": true
  }
}
```

`validated=true` в JSON сам по себе не является разрешением и не проверяется как доказательство. Одобрение задаётся внешним deployment digest. Zero-head и неподтверждённые/изменённые веса отклоняются.

Tokenizer сначала считывает полный текст с `truncation=false`. Если token count превышает `max_tokens`, scoring возвращает `unavailable` с явной причиной; молчаливого отбрасывания хвоста нет. Оконная агрегация не имитируется.

`backbone_status` отражает lazy/loaded/error загрузки, `inference_status` — not_run/healthy/error/rejected. После ошибки `risk_status=unavailable`; последующий успех очищает ошибочное inference-состояние.

## Проверки

```bash
pytest
node --test tests/test_ui.js
docker compose config --quiet
python -c "from fastapi.testclient import TestClient; from app import app; c=TestClient(app); print(c.get('/api/v1/health').json())"
```

UI-тесты используют только встроенный Node test runner, отдельных npm-зависимостей нет. CI использует hashed dev lock для pytest, запускает UI regression tests на Node 24 и отдельно выполняет Docker build. Для production ещё нужны независимая юридическая проверка правил и данных, обучение/калибровка/внешняя валидация и одобрение risk artifact, управление секретами, TLS/reverse proxy, rate limiting, мониторинг, резервное копирование и нагрузочные проверки.


## Dense kNN (`deepvk/USER-bge-m3`)

Production MVP поддерживает precomputed dense retrieval на `deepvk/USER-bge-m3`, revision `0cc6cfe48e260fb0474c753087a69369e88709ae`. Загрузка выполняется через `SentenceTransformer` на CPU по умолчанию, lazy и с `trust_remote_code=False`; batch size по умолчанию равен 4, token window — 1024 (`JUDICIAL_AI_DENSE_MAX_TOKENS`, CLI `--max-tokens`). Для построения index также поддерживается opt-in `--device cuda`; фактическое устройство сохраняется в generation manifest. Qwen не используется, appeal risk остаётся отдельным и выключенным.

### Построение index

После появления `corpus/processed/retrieval_cases.jsonl`:

```bash
# CPU по умолчанию
python tools/build_dense_index.py --batch-size 4 --max-tokens 1024 --checkpoint-batch-size 8

# Opt-in CUDA, если установлена совместимая CUDA-сборка PyTorch
python tools/build_dense_index.py --device cuda --batch-size 8 --max-tokens 1024 --checkpoint-batch-size 8
```

Завершённые поколения сохраняются в ignored `corpus/processed/dense-index-generations/<generation>/`. `model-card.json` — атомарно заменяемый pointer на активное поколение; index и его manifest всегда публикуются как завершённый комплект, предыдущий pointer сохраняется при ошибке. Legacy `dense-index.npz` + `model-card.json` поддерживается при первом resume и мигрируется на generations.

NPZ содержит строго `embeddings` и `text_hashes`: никаких `case_id`, narrative, punishment, appeal или outcome. Generation manifest хранит model/revision, размерности, агрегированные счётчики, preprocessing и SHA256 source/index, но не идентификаторы или тексты дел.

Новые embeddings сохраняются пакетами в `dense-index-work/<contract-sha256>/checkpoint.npz` с model/revision/preprocessing contract. Ошибка последнего документа или pointer publication не удаляет предыдущие пакеты и активное поколение. Последующий resume продолжает с checkpoint. Полный resume не загружает модель. Изменение любого requested preprocessing, включая `--max-tokens`, или model revision создаёт новый contract и пересобирает embeddings. Записи с дублирующимся narrative используют общий embedding.

Длинный narrative сначала токенизируется с `truncation=false`, затем полностью покрывается token windows с overlap 64. После decode каждый chunk повторно проверяется на token limit до encoder call. Chunk embeddings нормализуются, объединяются взвешенным по длине chunk усреднением и повторно нормализуются. Молчаливого отбрасывания хвоста нет. Записи без `article_parts` или с невалидными полями исключаются; их число сохраняется в `skipped_records` manifest и предупреждениях API.

### Runtime и API

При наличии processed JSONL backend использует его вместо `analysis.csv`; отсутствующий/невалидный processed source приводит к CSV fallback с warning. Сначала применяются прежние hard filters части статьи и process modes. Cosine вычисляется только после фильтров. Query кодируется той же pinned моделью lazy; базовый structured/unranked fallback сохраняется при отсутствии narrative, index, совместимых embeddings или модели.

При отсутствии numeric features distance равен scaled cosine `(1 - dot) / 2`. При наличии structured distance `d` используется понятная комбинация:

```text
structured_component = 1 - exp(-d)
distance = dense_weight * ((1 - dot) / 2) + (1 - dense_weight) * structured_component
```

`dense_weight=0.7` по умолчанию, `1.0` отключает structured вклад, а `0` немедленно возвращает исходный structured result без фильтрации записей или вызова encoder. Наказания, appeal и outcome не входят ни в vector payload, ни в similarity. Cohort statistics по-прежнему рассчитываются по всей hard-filtered cohort, включая записи без embedding; такие записи не участвуют в ranking и учитываются в warning.

`model_info.dense` сообщает repository, revision, `status`, `encoder_status`, `index_status`, `fallback_reason`, `distance_metric` и `dense_weight`. `model_info.knn.ranking_available` отражает фактический выбранный ranking/fallback. API analogues по-прежнему содержат только distance и punishment fields: `case_id`, source, narrative и appeal-метки не возвращаются.

Compose монтирует `${PROCESSED_DIR:-./corpus/processed}` в `/processed:ro`; model cache сохраняется в существующем volume. Переменные `JUDICIAL_AI_DENSE_*` приведены в `.env.example`.

### Независимая оценка качества

Актуальный benchmark — `corpus/processed/evaluation.md`: 79 подтверждённых пар, 75 независимых запросов после дедупликации. Для 19 известных сроков предсказание получено только для 10 (52,6%); для штрафов покрытие 0/5. MAE dense и structured совпадает — 19,2 месяца против 20,4 у cohort median. Это не доказывает преимущество dense или production-качество: нужны больше scored targets, внешняя/временная проверка и юридическая оценка аналогов.
