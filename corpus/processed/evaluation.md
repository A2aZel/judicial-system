# Независимая оценка kNN

## Протокол и итоговые counts

- Risk-записей на входе: **86**; после групповой дедупликации и проверки схемы оценено **83** запросов из **83** групп приговоров.
- Пропущено запросов: **3**; причины: article_part_missing=1, duplicate_verdict_group=2.
- Retrieval: **235** исходных, **172** валидных для production parser, **63** пропущено parser; после дедупликации осталось **172** документов.
- Непустая cohort: **78** из 83; пустая: **5**.
- Известные targets среди оценённых запросов: prison_months=**24**, fine_rub=**7**.
- Для каждого запроса исключены собственный hash текста, идентификатор дела и вся его verdict-группа. Дубликаты удалены также внутри candidate-side.
- Все методы используют одинаковую жёсткую cohort по первой перечисленной части статьи и процессуальным режимам; `null` не заменяется на `false`. Для ranking выбрано top_k=5.
- Cohort median использует всю cohort; остальные методы — медиану известных targets среди top-5. Неизвестный target не превращается в ноль. Coverage считается от запросов с известным фактическим target.
- Для fine_rub ошибки рассчитаны в пространстве `log1p(руб.)`; для prison_months — в месяцах. Cosine и combined distance являются только техническими мерами ranking, не юридической релевантностью.

## Counts по годам

Risk source — все подтверждённые записи; evaluated — после групповой дедупликации и проверки схемы. Retrieval source приведён до parser-фильтрации.

| Год | Risk source | Evaluated | С cohort | Prison known | Fine known | Retrieval source |
|---|---:|---:|---:|---:|---:|---:|
| 2017 | 0 | 0 | 0 | 0 | 0 | 23 |
| 2018 | 3 | 3 | 2 | 2 | 1 | 23 |
| 2019 | 11 | 11 | 11 | 1 | 1 | 33 |
| 2020 | 9 | 8 | 6 | 4 | 0 | 23 |
| 2021 | 2 | 1 | 1 | 0 | 0 | 7 |
| 2022 | 2 | 1 | 1 | 0 | 0 | 2 |
| 2023 | 2 | 2 | 2 | 1 | 0 | 8 |
| 2024 | 28 | 28 | 27 | 6 | 2 | 43 |
| 2025 | 15 | 15 | 15 | 5 | 1 | 53 |
| 2026 | 14 | 14 | 13 | 5 | 2 | 20 |

## Общие метрики

### prison_months, месяцы

| Метод | Запросов | Известных targets | Покрыто | Coverage | MAE | RMSE | Median AE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cohort median | 83 | 24 | 12 | 50.0% | 16.500 | 24.893 | 10.000 |
| Structured kNN | 83 | 24 | 11 | 45.8% | 15.545 | 23.012 | 12.000 |
| Dense kNN | 83 | 24 | 12 | 50.0% | 18.750 | 26.989 | 13.000 |
| Lexical BM25-like | 83 | 24 | 12 | 50.0% | 19.750 | 28.079 | 13.000 |
| Dense + structured | 83 | 24 | 11 | 45.8% | 15.273 | 22.994 | 12.000 |

### fine_rub, ошибки log1p(руб.)

| Метод | Запросов | Известных targets | Покрыто | Coverage | MAE | RMSE | Median AE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cohort median | 83 | 7 | 2 | 28.6% | 0.6315 | 0.8456 | 0.6315 |
| Structured kNN | 83 | 7 | 1 | 14.3% | 0.0690 | 0.0690 | 0.0690 |
| Dense kNN | 83 | 7 | 2 | 28.6% | 0.6315 | 0.8456 | 0.6315 |
| Lexical BM25-like | 83 | 7 | 2 | 28.6% | 0.6315 | 0.8456 | 0.6315 |
| Dense + structured | 83 | 7 | 1 | 14.3% | 0.0690 | 0.0690 | 0.0690 |

## Метрики по частям статьи и outcomes

### prison_months, месяцы

| Срез | Метод | Запросов | Известных targets | Покрыто | Coverage | MAE | RMSE | Median AE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| article_part:1 | Cohort median | 2 | 1 | 0 | 0.0% | — | — | — |
| article_part:1 | Structured kNN | 2 | 1 | 0 | 0.0% | — | — | — |
| article_part:1 | Dense kNN | 2 | 1 | 0 | 0.0% | — | — | — |
| article_part:1 | Lexical BM25-like | 2 | 1 | 0 | 0.0% | — | — | — |
| article_part:1 | Dense + structured | 2 | 1 | 0 | 0.0% | — | — | — |
| article_part:2 | Cohort median | 7 | 1 | 0 | 0.0% | — | — | — |
| article_part:2 | Structured kNN | 7 | 1 | 0 | 0.0% | — | — | — |
| article_part:2 | Dense kNN | 7 | 1 | 0 | 0.0% | — | — | — |
| article_part:2 | Lexical BM25-like | 7 | 1 | 0 | 0.0% | — | — | — |
| article_part:2 | Dense + structured | 7 | 1 | 0 | 0.0% | — | — | — |
| article_part:3 | Cohort median | 25 | 6 | 3 | 50.0% | 14.000 | 15.100 | 18.000 |
| article_part:3 | Structured kNN | 25 | 6 | 3 | 50.0% | 14.000 | 15.100 | 18.000 |
| article_part:3 | Dense kNN | 25 | 6 | 3 | 50.0% | 14.000 | 15.100 | 18.000 |
| article_part:3 | Lexical BM25-like | 25 | 6 | 3 | 50.0% | 14.000 | 15.100 | 18.000 |
| article_part:3 | Dense + structured | 25 | 6 | 3 | 50.0% | 14.000 | 15.100 | 18.000 |
| article_part:5 | Cohort median | 30 | 10 | 6 | 60.0% | 8.000 | 17.321 | 0.000 |
| article_part:5 | Structured kNN | 30 | 10 | 5 | 50.0% | 4.200 | 6.148 | 3.000 |
| article_part:5 | Dense kNN | 30 | 10 | 6 | 60.0% | 12.500 | 22.749 | 4.500 |
| article_part:5 | Lexical BM25-like | 30 | 10 | 6 | 60.0% | 14.500 | 25.249 | 6.000 |
| article_part:5 | Dense + structured | 30 | 10 | 5 | 50.0% | 3.600 | 6.000 | 0.000 |
| article_part:6 | Cohort median | 19 | 6 | 3 | 50.0% | 36.000 | 40.628 | 34.000 |
| article_part:6 | Structured kNN | 19 | 6 | 3 | 50.0% | 36.000 | 40.628 | 34.000 |
| article_part:6 | Dense kNN | 19 | 6 | 3 | 50.0% | 36.000 | 40.628 | 34.000 |
| article_part:6 | Lexical BM25-like | 19 | 6 | 3 | 50.0% | 36.000 | 40.628 | 34.000 |
| article_part:6 | Dense + structured | 19 | 6 | 3 | 50.0% | 36.000 | 40.628 | 34.000 |
| outcome:cancelled | Cohort median | 6 | 2 | 2 | 100.0% | 30.000 | 32.311 | 30.000 |
| outcome:cancelled | Structured kNN | 6 | 2 | 1 | 50.0% | 18.000 | 18.000 | 18.000 |
| outcome:cancelled | Dense kNN | 6 | 2 | 2 | 100.0% | 36.000 | 40.249 | 36.000 |
| outcome:cancelled | Lexical BM25-like | 6 | 2 | 2 | 100.0% | 39.000 | 44.294 | 39.000 |
| outcome:cancelled | Dense + structured | 6 | 2 | 1 | 50.0% | 18.000 | 18.000 | 18.000 |
| outcome:changed | Cohort median | 44 | 12 | 5 | 41.7% | 13.200 | 26.967 | 0.000 |
| outcome:changed | Structured kNN | 44 | 12 | 5 | 41.7% | 16.200 | 27.528 | 6.000 |
| outcome:changed | Dense kNN | 44 | 12 | 5 | 41.7% | 16.200 | 27.528 | 6.000 |
| outcome:changed | Lexical BM25-like | 44 | 12 | 5 | 41.7% | 17.400 | 27.659 | 6.000 |
| outcome:changed | Dense + structured | 44 | 12 | 5 | 41.7% | 15.600 | 27.495 | 6.000 |
| outcome:unchanged | Cohort median | 33 | 10 | 5 | 50.0% | 14.400 | 18.504 | 14.000 |
| outcome:unchanged | Structured kNN | 33 | 10 | 5 | 50.0% | 14.400 | 18.504 | 14.000 |
| outcome:unchanged | Dense kNN | 33 | 10 | 5 | 50.0% | 14.400 | 18.504 | 14.000 |
| outcome:unchanged | Lexical BM25-like | 33 | 10 | 5 | 50.0% | 14.400 | 18.504 | 14.000 |
| outcome:unchanged | Dense + structured | 33 | 10 | 5 | 50.0% | 14.400 | 18.504 | 14.000 |

### fine_rub, ошибки log1p(руб.)

| Срез | Метод | Запросов | Известных targets | Покрыто | Coverage | MAE | RMSE | Median AE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| article_part:1 | Cohort median | 2 | 0 | 0 | — | — | — | — |
| article_part:1 | Structured kNN | 2 | 0 | 0 | — | — | — | — |
| article_part:1 | Dense kNN | 2 | 0 | 0 | — | — | — | — |
| article_part:1 | Lexical BM25-like | 2 | 0 | 0 | — | — | — | — |
| article_part:1 | Dense + structured | 2 | 0 | 0 | — | — | — | — |
| article_part:2 | Cohort median | 7 | 0 | 0 | — | — | — | — |
| article_part:2 | Structured kNN | 7 | 0 | 0 | — | — | — | — |
| article_part:2 | Dense kNN | 7 | 0 | 0 | — | — | — | — |
| article_part:2 | Lexical BM25-like | 7 | 0 | 0 | — | — | — | — |
| article_part:2 | Dense + structured | 7 | 0 | 0 | — | — | — | — |
| article_part:3 | Cohort median | 25 | 2 | 1 | 50.0% | 0.0690 | 0.0690 | 0.0690 |
| article_part:3 | Structured kNN | 25 | 2 | 1 | 50.0% | 0.0690 | 0.0690 | 0.0690 |
| article_part:3 | Dense kNN | 25 | 2 | 1 | 50.0% | 0.0690 | 0.0690 | 0.0690 |
| article_part:3 | Lexical BM25-like | 25 | 2 | 1 | 50.0% | 0.0690 | 0.0690 | 0.0690 |
| article_part:3 | Dense + structured | 25 | 2 | 1 | 50.0% | 0.0690 | 0.0690 | 0.0690 |
| article_part:5 | Cohort median | 30 | 2 | 1 | 50.0% | 1.1939 | 1.1939 | 1.1939 |
| article_part:5 | Structured kNN | 30 | 2 | 0 | 0.0% | — | — | — |
| article_part:5 | Dense kNN | 30 | 2 | 1 | 50.0% | 1.1939 | 1.1939 | 1.1939 |
| article_part:5 | Lexical BM25-like | 30 | 2 | 1 | 50.0% | 1.1939 | 1.1939 | 1.1939 |
| article_part:5 | Dense + structured | 30 | 2 | 0 | 0.0% | — | — | — |
| article_part:6 | Cohort median | 19 | 3 | 0 | 0.0% | — | — | — |
| article_part:6 | Structured kNN | 19 | 3 | 0 | 0.0% | — | — | — |
| article_part:6 | Dense kNN | 19 | 3 | 0 | 0.0% | — | — | — |
| article_part:6 | Lexical BM25-like | 19 | 3 | 0 | 0.0% | — | — | — |
| article_part:6 | Dense + structured | 19 | 3 | 0 | 0.0% | — | — | — |
| outcome:cancelled | Cohort median | 6 | 1 | 1 | 100.0% | 1.1939 | 1.1939 | 1.1939 |
| outcome:cancelled | Structured kNN | 6 | 1 | 0 | 0.0% | — | — | — |
| outcome:cancelled | Dense kNN | 6 | 1 | 1 | 100.0% | 1.1939 | 1.1939 | 1.1939 |
| outcome:cancelled | Lexical BM25-like | 6 | 1 | 1 | 100.0% | 1.1939 | 1.1939 | 1.1939 |
| outcome:cancelled | Dense + structured | 6 | 1 | 0 | 0.0% | — | — | — |
| outcome:changed | Cohort median | 44 | 4 | 0 | 0.0% | — | — | — |
| outcome:changed | Structured kNN | 44 | 4 | 0 | 0.0% | — | — | — |
| outcome:changed | Dense kNN | 44 | 4 | 0 | 0.0% | — | — | — |
| outcome:changed | Lexical BM25-like | 44 | 4 | 0 | 0.0% | — | — | — |
| outcome:changed | Dense + structured | 44 | 4 | 0 | 0.0% | — | — | — |
| outcome:unchanged | Cohort median | 33 | 2 | 1 | 50.0% | 0.0690 | 0.0690 | 0.0690 |
| outcome:unchanged | Structured kNN | 33 | 2 | 1 | 50.0% | 0.0690 | 0.0690 | 0.0690 |
| outcome:unchanged | Dense kNN | 33 | 2 | 1 | 50.0% | 0.0690 | 0.0690 | 0.0690 |
| outcome:unchanged | Lexical BM25-like | 33 | 2 | 1 | 50.0% | 0.0690 | 0.0690 | 0.0690 |
| outcome:unchanged | Dense + structured | 33 | 2 | 1 | 50.0% | 0.0690 | 0.0690 | 0.0690 |

## Техническая проверка ranking

| Метод | Запросов с neighbors | Возвращено neighbors | Distance non-null | Distance null | Self exclusions нарушено | Cohort filter нарушено |
|---|---:|---:|---:|---:|---:|---:|
| Structured kNN | 78 | 286 | 286 | 0 | 0 | 0 |
| Dense kNN | 78 | 286 | 286 | 0 | 0 | 0 |
| Lexical BM25-like | 78 | 286 | 286 | 0 | 0 | 0 |
| Dense + structured | 78 | 286 | 286 | 0 | 0 | 0 |

- Structured без активных числовых признаков (unranked fallback): **0** запросов; у такого списка допустимы `distance=null`.
- Dense fallback: **{"empty_cohort": 5}**; dense+structured: **{"empty_cohort": 5}**.
- Отсутствующих embeddings внутри cohort: **0**.

## Сравнение и ограничения

- По prison_months MAE dense=18.750, structured=15.545, cohort median=16.500; это описание малой выборки, а не доказанное преимущество.
- По fine_rub MAE dense=0.6315, structured=0.0690, cohort median=0.6315; это описание малой выборки, а не доказанное преимущество.
- Выборка мала и нерепрезентативна: только подтверждённые полные пары одного пилота; внешней, временной или региональной проверки нет. Число известных targets существенно меньше числа запросов, поэтому RMSE неустойчив к отдельным делам.
- Пустые cohort и срезы с малым `n_known` означают отсутствие доказанного покрытия, а не нулевую ошибку. Прочерк в метрике означает отсутствие наблюдений.
- Одинаковые candidate-side и hard filters делают сравнение честным технически, но не устраняют ошибки извлечения признаков и targets из текстов.
- Dense+structured использует заранее заданный production вес 0.7; подбор веса по этой выборке не проводился. Доверительные интервалы и тесты значимости не дают надёжного вывода при таких counts.
- Локальный review pack с идентификаторами и excerpts предназначен только для ручной проверки lead. Он не публикуется; автоматическая оценка не заменяет эту проверку.
- **Production quality не доказано**: нужны больше независимых дел с известными targets, групповая внешняя/временная проверка и ручная оценка юридической уместности аналогов.
