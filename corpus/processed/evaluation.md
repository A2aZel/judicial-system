# Независимая оценка kNN

## Протокол и итоговые counts

- Risk-записей на входе: **79**; после групповой дедупликации и проверки схемы оценено **75** запросов из **75** групп приговоров.
- Пропущено запросов: **4**; причины: article_part_missing=1, duplicate_verdict_group=3.
- Retrieval: **205** исходных, **142** валидных для production parser, **63** пропущено parser; после дедупликации осталось **142** документов.
- Непустая cohort: **64** из 75; пустая: **11**.
- Известные targets среди оценённых запросов: prison_months=**19**, fine_rub=**5**.
- Для каждого запроса исключены собственный hash текста, идентификатор дела и вся его verdict-группа. Дубликаты удалены также внутри candidate-side.
- Все методы используют одинаковую жёсткую cohort по первой перечисленной части статьи и процессуальным режимам; `null` не заменяется на `false`. Для ranking выбрано top_k=5.
- Cohort median использует всю cohort; остальные методы — медиану известных targets среди top-5. Неизвестный target не превращается в ноль. Coverage считается от запросов с известным фактическим target.
- Для fine_rub ошибки рассчитаны в пространстве `log1p(руб.)`; для prison_months — в месяцах. Cosine и combined distance являются только техническими мерами ranking, не юридической релевантностью.

## Counts по годам

Risk source — все подтверждённые записи; evaluated — после групповой дедупликации и проверки схемы. Retrieval source приведён до parser-фильтрации.

| Год | Risk source | Evaluated | С cohort | Prison known | Fine known | Retrieval source |
|---|---:|---:|---:|---:|---:|---:|
| 2017 | 0 | 0 | 0 | 0 | 0 | 23 |
| 2018 | 3 | 3 | 1 | 2 | 1 | 23 |
| 2019 | 11 | 11 | 11 | 1 | 1 | 30 |
| 2020 | 8 | 7 | 5 | 4 | 0 | 21 |
| 2021 | 1 | 0 | 0 | 0 | 0 | 5 |
| 2022 | 2 | 1 | 1 | 0 | 0 | 2 |
| 2023 | 2 | 2 | 2 | 0 | 0 | 6 |
| 2024 | 27 | 26 | 22 | 5 | 2 | 35 |
| 2025 | 13 | 13 | 12 | 3 | 1 | 41 |
| 2026 | 12 | 12 | 10 | 4 | 0 | 19 |

## Общие метрики

### prison_months, месяцы

| Метод | Запросов | Известных targets | Покрыто | Coverage | MAE | RMSE | Median AE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cohort median | 75 | 19 | 10 | 52.6% | 20.400 | 26.635 | 16.000 |
| Structured kNN | 75 | 19 | 10 | 52.6% | 19.200 | 25.247 | 16.000 |
| Dense kNN | 75 | 19 | 10 | 52.6% | 19.200 | 25.247 | 16.000 |
| Lexical BM25-like | 75 | 19 | 10 | 52.6% | 20.400 | 26.635 | 16.000 |
| Dense + structured | 75 | 19 | 10 | 52.6% | 19.200 | 25.247 | 16.000 |

### fine_rub, ошибки log1p(руб.)

| Метод | Запросов | Известных targets | Покрыто | Coverage | MAE | RMSE | Median AE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cohort median | 75 | 5 | 0 | 0.0% | — | — | — |
| Structured kNN | 75 | 5 | 0 | 0.0% | — | — | — |
| Dense kNN | 75 | 5 | 0 | 0.0% | — | — | — |
| Lexical BM25-like | 75 | 5 | 0 | 0.0% | — | — | — |
| Dense + structured | 75 | 5 | 0 | 0.0% | — | — | — |

## Метрики по частям статьи и outcomes

### prison_months, месяцы

| Срез | Метод | Запросов | Известных targets | Покрыто | Coverage | MAE | RMSE | Median AE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| article_part:1 | Cohort median | 1 | 1 | 0 | 0.0% | — | — | — |
| article_part:1 | Structured kNN | 1 | 1 | 0 | 0.0% | — | — | — |
| article_part:1 | Dense kNN | 1 | 1 | 0 | 0.0% | — | — | — |
| article_part:1 | Lexical BM25-like | 1 | 1 | 0 | 0.0% | — | — | — |
| article_part:1 | Dense + structured | 1 | 1 | 0 | 0.0% | — | — | — |
| article_part:2 | Cohort median | 7 | 1 | 0 | 0.0% | — | — | — |
| article_part:2 | Structured kNN | 7 | 1 | 0 | 0.0% | — | — | — |
| article_part:2 | Dense kNN | 7 | 1 | 0 | 0.0% | — | — | — |
| article_part:2 | Lexical BM25-like | 7 | 1 | 0 | 0.0% | — | — | — |
| article_part:2 | Dense + structured | 7 | 1 | 0 | 0.0% | — | — | — |
| article_part:3 | Cohort median | 21 | 3 | 2 | 66.7% | 18.000 | 18.000 | 18.000 |
| article_part:3 | Structured kNN | 21 | 3 | 2 | 66.7% | 18.000 | 18.000 | 18.000 |
| article_part:3 | Dense kNN | 21 | 3 | 2 | 66.7% | 18.000 | 18.000 | 18.000 |
| article_part:3 | Lexical BM25-like | 21 | 3 | 2 | 66.7% | 18.000 | 18.000 | 18.000 |
| article_part:3 | Dense + structured | 21 | 3 | 2 | 66.7% | 18.000 | 18.000 | 18.000 |
| article_part:5 | Cohort median | 27 | 9 | 5 | 55.6% | 12.000 | 17.286 | 6.000 |
| article_part:5 | Structured kNN | 27 | 9 | 5 | 55.6% | 9.600 | 12.442 | 6.000 |
| article_part:5 | Dense kNN | 27 | 9 | 5 | 55.6% | 9.600 | 12.442 | 6.000 |
| article_part:5 | Lexical BM25-like | 27 | 9 | 5 | 55.6% | 12.000 | 17.286 | 6.000 |
| article_part:5 | Dense + structured | 27 | 9 | 5 | 55.6% | 9.600 | 12.442 | 6.000 |
| article_part:6 | Cohort median | 19 | 5 | 3 | 60.0% | 36.000 | 40.628 | 34.000 |
| article_part:6 | Structured kNN | 19 | 5 | 3 | 60.0% | 36.000 | 40.628 | 34.000 |
| article_part:6 | Dense kNN | 19 | 5 | 3 | 60.0% | 36.000 | 40.628 | 34.000 |
| article_part:6 | Lexical BM25-like | 19 | 5 | 3 | 60.0% | 36.000 | 40.628 | 34.000 |
| article_part:6 | Dense + structured | 19 | 5 | 3 | 60.0% | 36.000 | 40.628 | 34.000 |
| outcome:cancelled | Cohort median | 7 | 2 | 2 | 100.0% | 27.000 | 28.460 | 27.000 |
| outcome:cancelled | Structured kNN | 7 | 2 | 2 | 100.0% | 21.000 | 21.213 | 21.000 |
| outcome:cancelled | Dense kNN | 7 | 2 | 2 | 100.0% | 21.000 | 21.213 | 21.000 |
| outcome:cancelled | Lexical BM25-like | 7 | 2 | 2 | 100.0% | 27.000 | 28.460 | 27.000 |
| outcome:cancelled | Dense + structured | 7 | 2 | 2 | 100.0% | 21.000 | 21.213 | 21.000 |
| outcome:changed | Cohort median | 38 | 10 | 5 | 50.0% | 16.800 | 27.561 | 6.000 |
| outcome:changed | Structured kNN | 38 | 10 | 5 | 50.0% | 16.800 | 27.561 | 6.000 |
| outcome:changed | Dense kNN | 38 | 10 | 5 | 50.0% | 16.800 | 27.561 | 6.000 |
| outcome:changed | Lexical BM25-like | 38 | 10 | 5 | 50.0% | 16.800 | 27.561 | 6.000 |
| outcome:changed | Dense + structured | 38 | 10 | 5 | 50.0% | 16.800 | 27.561 | 6.000 |
| outcome:unchanged | Cohort median | 30 | 7 | 3 | 42.9% | 22.000 | 23.636 | 18.000 |
| outcome:unchanged | Structured kNN | 30 | 7 | 3 | 42.9% | 22.000 | 23.636 | 18.000 |
| outcome:unchanged | Dense kNN | 30 | 7 | 3 | 42.9% | 22.000 | 23.636 | 18.000 |
| outcome:unchanged | Lexical BM25-like | 30 | 7 | 3 | 42.9% | 22.000 | 23.636 | 18.000 |
| outcome:unchanged | Dense + structured | 30 | 7 | 3 | 42.9% | 22.000 | 23.636 | 18.000 |

### fine_rub, ошибки log1p(руб.)

| Срез | Метод | Запросов | Известных targets | Покрыто | Coverage | MAE | RMSE | Median AE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| article_part:1 | Cohort median | 1 | 0 | 0 | — | — | — | — |
| article_part:1 | Structured kNN | 1 | 0 | 0 | — | — | — | — |
| article_part:1 | Dense kNN | 1 | 0 | 0 | — | — | — | — |
| article_part:1 | Lexical BM25-like | 1 | 0 | 0 | — | — | — | — |
| article_part:1 | Dense + structured | 1 | 0 | 0 | — | — | — | — |
| article_part:2 | Cohort median | 7 | 0 | 0 | — | — | — | — |
| article_part:2 | Structured kNN | 7 | 0 | 0 | — | — | — | — |
| article_part:2 | Dense kNN | 7 | 0 | 0 | — | — | — | — |
| article_part:2 | Lexical BM25-like | 7 | 0 | 0 | — | — | — | — |
| article_part:2 | Dense + structured | 7 | 0 | 0 | — | — | — | — |
| article_part:3 | Cohort median | 21 | 0 | 0 | — | — | — | — |
| article_part:3 | Structured kNN | 21 | 0 | 0 | — | — | — | — |
| article_part:3 | Dense kNN | 21 | 0 | 0 | — | — | — | — |
| article_part:3 | Lexical BM25-like | 21 | 0 | 0 | — | — | — | — |
| article_part:3 | Dense + structured | 21 | 0 | 0 | — | — | — | — |
| article_part:5 | Cohort median | 27 | 2 | 0 | 0.0% | — | — | — |
| article_part:5 | Structured kNN | 27 | 2 | 0 | 0.0% | — | — | — |
| article_part:5 | Dense kNN | 27 | 2 | 0 | 0.0% | — | — | — |
| article_part:5 | Lexical BM25-like | 27 | 2 | 0 | 0.0% | — | — | — |
| article_part:5 | Dense + structured | 27 | 2 | 0 | 0.0% | — | — | — |
| article_part:6 | Cohort median | 19 | 3 | 0 | 0.0% | — | — | — |
| article_part:6 | Structured kNN | 19 | 3 | 0 | 0.0% | — | — | — |
| article_part:6 | Dense kNN | 19 | 3 | 0 | 0.0% | — | — | — |
| article_part:6 | Lexical BM25-like | 19 | 3 | 0 | 0.0% | — | — | — |
| article_part:6 | Dense + structured | 19 | 3 | 0 | 0.0% | — | — | — |
| outcome:cancelled | Cohort median | 7 | 1 | 0 | 0.0% | — | — | — |
| outcome:cancelled | Structured kNN | 7 | 1 | 0 | 0.0% | — | — | — |
| outcome:cancelled | Dense kNN | 7 | 1 | 0 | 0.0% | — | — | — |
| outcome:cancelled | Lexical BM25-like | 7 | 1 | 0 | 0.0% | — | — | — |
| outcome:cancelled | Dense + structured | 7 | 1 | 0 | 0.0% | — | — | — |
| outcome:changed | Cohort median | 38 | 4 | 0 | 0.0% | — | — | — |
| outcome:changed | Structured kNN | 38 | 4 | 0 | 0.0% | — | — | — |
| outcome:changed | Dense kNN | 38 | 4 | 0 | 0.0% | — | — | — |
| outcome:changed | Lexical BM25-like | 38 | 4 | 0 | 0.0% | — | — | — |
| outcome:changed | Dense + structured | 38 | 4 | 0 | 0.0% | — | — | — |
| outcome:unchanged | Cohort median | 30 | 0 | 0 | — | — | — | — |
| outcome:unchanged | Structured kNN | 30 | 0 | 0 | — | — | — | — |
| outcome:unchanged | Dense kNN | 30 | 0 | 0 | — | — | — | — |
| outcome:unchanged | Lexical BM25-like | 30 | 0 | 0 | — | — | — | — |
| outcome:unchanged | Dense + structured | 30 | 0 | 0 | — | — | — | — |

## Техническая проверка ranking

| Метод | Запросов с neighbors | Возвращено neighbors | Distance non-null | Distance null | Self exclusions нарушено | Cohort filter нарушено |
|---|---:|---:|---:|---:|---:|---:|
| Structured kNN | 64 | 189 | 189 | 0 | 0 | 0 |
| Dense kNN | 64 | 189 | 189 | 0 | 0 | 0 |
| Lexical BM25-like | 64 | 189 | 189 | 0 | 0 | 0 |
| Dense + structured | 64 | 189 | 189 | 0 | 0 | 0 |

- Structured без активных числовых признаков (unranked fallback): **0** запросов; у такого списка допустимы `distance=null`.
- Dense fallback: **{"empty_cohort": 11}**; dense+structured: **{"empty_cohort": 11}**.
- Отсутствующих embeddings внутри cohort: **0**.

## Сравнение и ограничения

- По prison_months MAE dense=19.200, structured=19.200, cohort median=20.400; это описание малой выборки, а не доказанное преимущество.
- По fine_rub MAE dense=—, structured=—, cohort median=—; это описание малой выборки, а не доказанное преимущество.
- Выборка мала и нерепрезентативна: только подтверждённые полные пары одного пилота; внешней, временной или региональной проверки нет. Число известных targets существенно меньше числа запросов, поэтому RMSE неустойчив к отдельным делам.
- Пустые cohort и срезы с малым `n_known` означают отсутствие доказанного покрытия, а не нулевую ошибку. Прочерк в метрике означает отсутствие наблюдений.
- Одинаковые candidate-side и hard filters делают сравнение честным технически, но не устраняют ошибки извлечения признаков и targets из текстов.
- Dense+structured использует заранее заданный production вес 0.7; подбор веса по этой выборке не проводился. Доверительные интервалы и тесты значимости не дают надёжного вывода при таких counts.
- Локальный review pack с идентификаторами и excerpts предназначен только для ручной проверки lead. Он не публикуется; автоматическая оценка не заменяет эту проверку.
- **Production quality не доказано**: нужны больше независимых дел с известными targets, групповая внешняя/временная проверка и ручная оценка юридической уместности аналогов.
