# Версионируемый dataset для kNN и анализа риска апелляции

`schema_version=1`. Dataset-only артефакт, не production-модель и не оценка production readiness.

## Файлы

- `cases.jsonl` — все записи с отдельными полями первой инстанции и апелляции.
- `retrieval_cases.jsonl` — только пригодные для kNN тексты первой инстанции, без labels и апелляционного текста.
- `risk_examples.jsonl` — только `full_pair=true`: факты и наказание первой инстанции отдельно от `outcome`.
- `article_examples.jsonl` — классификация статьи/части по имеющимся меткам; отсутствующие части не выдумываются.
- `manifest.json` — схема и агрегаты, без перечней ID и PII.
- `dataset-card.json` — dataset-only intended use и правила против утечки.
- `raw/id-map.jsonl` — локальное отображение псевдонимов в исходные ключи для resume.
- `raw/texts/` — локальные копии текстов с псевдонимами.
- `raw/build-state.json` — hash исходников и manifest для возобновления.

`cases.jsonl`, `retrieval_cases.jsonl`, `risk_examples.jsonl`, `article_examples.jsonl` и `raw/` содержат тексты и могут содержать PII. Они не предназначены для commit или публикации.

## Схема записи

Обязательные поля: `schema_version`, `case_id`, `verdict_group_id`, `source`, `source_year`, `article_parts`, `other_articles`, `process`, `facts`, `punishment`, `appeal`, `quality`, `split`.

- `case_id` — детерминированный псевдоним отдельного appellate event. Несколько апелляций одного приговора не сливаются.
- `verdict_group_id` — группа одного приговора для group split. Вычисляется от hash текста и canonical identity без source, поэтому дубликаты из разных источников не попадают в разные splits.
- `article_parts` и `process` — только квалификация и процессуальные сведения первой инстанции. Апелляционная переквалификация хранится отдельно в `appeal.outcome_article_parts` и не становится feature.
- `facts.narrative` — только текст приговора первой инстанции. Апелляционный текст в features не входит.
- `punishment` — наказание первой инстанции, отдельно от `appeal.outcome`.
- `appeal.full_pair` — подтверждение двух полных текстов, их актуальных SHA256 и identity московским validator. Для неполных пар `false`.
- `appeal.outcome`: `unchanged` / `changed` / `cancelled` / `unknown`. `unknown` не считается negative.
- `quality.text_coverage`, `redaction_flags`, `parser_status`, `identity_errors` — фактическое покрытие и ограничения обработки. Тексты с ошибочной identity не попадают в retrieval.
- Nullable-схема: неизвестные значения — `null` или пустые списки, без догадок. Отсутствие сведения о process flag не превращается в `False`; `True` ставится только при подтверждении из текста или structured metadata.

## Anti-leakage правила

1. `retrieval_cases` не содержит `appeal`, labels или апелляционный текст.
2. `risk_examples` включает только подтверждённые полные пары; неподтверждённые московские кандидаты не используются как labels.
3. Признаки и наказание первой инстанции отделены от исхода апелляции.
4. `unknown` не превращается в negative class.
5. Group split держит все апелляции одного приговора в одной группе.

## Resume

Из корня проекта:

```bash
python tools/build_training_dataset.py

# Принудительная пересборка без изменения детерминированных ID.
python tools/build_training_dataset.py --force

# Опциональное детерминированное групповое разбиение train/validation/test.
python tools/build_training_dataset.py --split
```

Повторный запуск без изменения исходников возвращает сохранённый manifest только после проверки schema, counts и SHA256 всех outputs, model-card, id-map и текстовых артефактов. Режим split входит в build-state: смена `unassigned → split` и обратно обязательно пересобирает файлы. Пустые или повреждённые outputs пересобираются. Псевдонимы детерминированы и не зависят от времени или случайных голов BERT.

## Источники и ограничения

- Moscow: 207 кандидатов, risk labels только для 50 подтверждённых полных пар.
- SudAct: 129 приговоров и 37 апелляций. Для них используются тексты и structured CSV; неподтверждённые связи не становятся risk examples.
- Regional: в текущем `corpus/regional/raw` есть только robots/access. Исторические 4 полные пары упомянуты в прежнем пилоте, но их тексты отсутствуют, поэтому не включены. В manifest это явно отмечено как `regional_historical_unavailable=4`.

Датасет не претендует на полноту годовой практики, отсутствие смещения выборки или готовность к production. Извлечённые признаки требуют отдельной проверки качества и обезличивания перед дообучением.

## Тесты

```bash
python -m pytest tests/test_training_dataset.py -q
```

Синтетические тесты проверяют детерминизм, несколько апелляций одного приговора, anti-leakage, unknown != negative, group split, покрытие источников, схему manifest и resume.
