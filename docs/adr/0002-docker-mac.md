# ADR 0002: CPU Docker для Apple Silicon и Intel Mac

Дата: 2026-11-23
Статус: принято

## Контекст

Существующий Dockerfile принудительно выбирал Linux amd64. На Apple Silicon это
требовало эмуляции, а единственный lock зависимостей содержал amd64 wheel PyTorch.
Нужна переносимая упаковка существующего structured/dense kNN без изменения UI/API,
добавления Qwen или публикации локального судебного корпуса. Целевая конфигурация
включает Mac с 8 ГБ RAM; замеров latency и peak RAM на Mac пока нет.

## Решение

- Обе стадии Dockerfile используют один immutable multi-platform index
  `python:3.12.10-slim@sha256:fd95fa221297a88e1cf49c55ec1828edd7c5a428187e67b5d1805692d11588db`.
  Выбор Linux manifest и архитектуры выполняет Docker.
- `requirements-lock.txt` сохраняет amd64 CPU runtime;
  `requirements-arm64-lock.txt` отдельно закрепляет arm64 wheels. Docker выбирает
  lock по `TARGETARCH`, установка требует hashes. CUDA/NVIDIA пакеты не добавляются.
- `compose.mac.yaml` — самостоятельный профиль: loopback-порт, readonly root/data,
  пользователь без root, отдельный model cache, 4 ГБ / 2 CPU, dense batch size 1,
  appeal risk выключен. Linux CPU используется на обеих Mac-архитектурах; MPS/Metal
  не является backend этого контейнера.
- В образ попадает только явно разрешённый runtime и синтетический CSV. Реальные
  данные хранятся в host `data/processed/`, включая полные index generations, и
  монтируются read-only в `/data`. Веса Hugging Face остаются в отдельном volume
  или явно выбранной host-папке. Qwen отсутствует и не скачивается.
- API, UI и правила доступа сохраняются: локальный development без ключа;
  для production нужны существующий обязательный Bearer key, TLS/reverse proxy
  и осознанное изменение bind address. Mac-профиль сам не открывает порт в сеть.
- `.github/workflows/docker-mac.yml` использует native `ubuntu-24.04` (amd64) и
  `ubuntu-24.04-arm` (arm64). Каждая сборка загружается локально и проходит HTTP
  smoke с реальным POST анализа синтетических записей и проверку CPU-only torch.
  Затем именно этот image публикуется с per-architecture SHA tag; между jobs
  передаются только строго проверенные registry digests.
- После успеха обеих архитектур workflow собирает manifest и проверяет совпадение
  платформ/digests. Только затем обновляет `ghcr.io/a2azel/judicial-system:mac8gb`.
  `sha-<полный commit SHA>` закрепляет сборку; `latest` не публикуется. Используется
  `GITHUB_TOKEN` с `packages: write`, без новых секретов. Владелец отдельно проверяет
  public visibility GHCR package для анонимного скачивания.

## Рассмотренные варианты

Принудительный amd64 с QEMU оставил бы один lock, но увеличил бы время сборки и
создал постоянную эмуляцию на Apple Silicon. Выбраны два нативных CPU lock и два
native CI runner: каждая архитектура проверяется в собственном Linux runtime.

Включение реального корпуса и весов в image упростило бы первый dense-запрос, но
связало бы релиз кода с обновлением данных и публиковало бы потенциально персональные
материалы. Выбрано отдельное хранение данных и model cache; фикстура явно обозначена
как синтетическая.

## Последствия и проверка

Структурированный kNN доступен на фикстуре без загрузок моделей. Dense retrieval
работает по прежнему контракту при наличии совместимого processed JSONL/index и
непустой query narrative; первая операция может скачать pinned encoder. CI smoke
не доказывает качество аналогов или прохождение dense inference на Mac.

Docker VM бюджет 4–5 ГБ и service cap 4 ГБ — начальные настройки для машины с
8 ГБ RAM, не гарантия вместимости dense-модели. При OOM доступно существующее
`JUDICIAL_AI_DENSE_ENABLED=false`; для dense нужны замеры на целевом Mac.
Изменение зависимостей требует повторного разрешения обоих lock и проверки обеих
архитектур. Real corpus/cache остаются локальными и не должны попадать в registry.

## Источники

- [Docker: multi-platform images in GitHub Actions](https://docs.docker.com/build/ci/github-actions/multi-platform/)
- [Docker: creating a multi-platform manifest](https://docs.docker.com/reference/cli/docker/buildx/imagetools/create/)
- [GitHub: Container registry, access and initial visibility](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry)
