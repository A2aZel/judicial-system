# Независимая оценка kNN

## Протокол и итоговые counts

- Risk-записей на входе: **77**; после групповой дедупликации и проверки схемы оценено **74** запросов из **74** групп приговоров.
- Пропущено запросов: **3**; причины: article_part_missing=1, duplicate_verdict_group=2.
- Retrieval: **229** исходных, **166** валидных для production parser, **63** пропущено parser; после дедупликации осталось **166** документов.
- Непустая cohort: **69** из 74; пустая: **5**.
- Известные targets среди оценённых запросов: prison_months=**21**, fine_rub=**6**.
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
| 2020 | 8 | 7 | 5 | 4 | 0 | 22 |
| 2021 | 1 | 0 | 0 | 0 | 0 | 6 |
| 2022 | 2 | 1 | 1 | 0 | 0 | 2 |
| 2023 | 1 | 1 | 1 | 0 | 0 | 7 |
| 2024 | 25 | 25 | 24 | 5 | 2 | 42 |
| 2025 | 12 | 12 | 12 | 3 | 1 | 52 |
| 2026 | 14 | 14 | 13 | 6 | 1 | 19 |

## Общие метрики

### prison_months, месяцы

| Метод | Запросов | Известных targets | Покрыто | Coverage | MAE | RMSE | Median AE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cohort median | 74 | 21 | 12 | 57.1% | 17.500 | 25.133 | 13.000 |
| Structured kNN | 74 | 21 | 11 | 52.4% | 16.636 | 23.295 | 12.000 |
| Dense kNN | 74 | 21 | 12 | 57.1% | 19.750 | 27.211 | 13.000 |
| Lexical BM25-like | 74 | 21 | 12 | 57.1% | 20.750 | 28.292 | 13.000 |
| Dense + structured | 74 | 21 | 11 | 52.4% | 16.364 | 23.277 | 12.000 |

### fine_rub, ошибки log1p(руб.)

| Метод | Запросов | Известных targets | Покрыто | Coverage | MAE | RMSE | Median AE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cohort median | 74 | 6 | 2 | 33.3% | 0.6315 | 0.8456 | 0.6315 |
| Structured kNN | 74 | 6 | 1 | 16.7% | 0.0690 | 0.0690 | 0.0690 |
| Dense kNN | 74 | 6 | 2 | 33.3% | 0.6315 | 0.8456 | 0.6315 |
| Lexical BM25-like | 74 | 6 | 2 | 33.3% | 0.6315 | 0.8456 | 0.6315 |
| Dense + structured | 74 | 6 | 1 | 16.7% | 0.0690 | 0.0690 | 0.0690 |

## Метрики по частям статьи и outcomes

### prison_months, месяцы

| Срез | Метод | Запросов | Известных targets | Покрыто | Coverage | MAE | RMSE | Median AE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| article_part:1 | Cohort median | 1 | 1 | 0 | 0.0% | — | — | — |
| article_part:1 | Structured kNN | 1 | 1 | 0 | 0.0% | — | — | — |
| article_part:1 | Dense kNN | 1 | 1 | 0 | 0.0% | — | — | — |
| article_part:1 | Lexical BM25-like | 1 | 1 | 0 | 0.0% | — | — | — |
| article_part:1 | Dense + structured | 1 | 1 | 0 | 0.0% | — | — | — |
| article_part:2 | Cohort median | 8 | 2 | 1 | 50.0% | 12.000 | 12.000 | 12.000 |
| article_part:2 | Structured kNN | 8 | 2 | 1 | 50.0% | 12.000 | 12.000 | 12.000 |
| article_part:2 | Dense kNN | 8 | 2 | 1 | 50.0% | 12.000 | 12.000 | 12.000 |
| article_part:2 | Lexical BM25-like | 8 | 2 | 1 | 50.0% | 12.000 | 12.000 | 12.000 |
| article_part:2 | Dense + structured | 8 | 2 | 1 | 50.0% | 12.000 | 12.000 | 12.000 |
| article_part:3 | Cohort median | 21 | 4 | 3 | 75.0% | 14.000 | 15.100 | 18.000 |
| article_part:3 | Structured kNN | 21 | 4 | 3 | 75.0% | 14.000 | 15.100 | 18.000 |
| article_part:3 | Dense kNN | 21 | 4 | 3 | 75.0% | 14.000 | 15.100 | 18.000 |
| article_part:3 | Lexical BM25-like | 21 | 4 | 3 | 75.0% | 14.000 | 15.100 | 18.000 |
| article_part:3 | Dense + structured | 21 | 4 | 3 | 75.0% | 14.000 | 15.100 | 18.000 |
| article_part:5 | Cohort median | 26 | 9 | 5 | 55.6% | 9.600 | 18.974 | 0.000 |
| article_part:5 | Structured kNN | 26 | 9 | 4 | 44.4% | 5.250 | 6.874 | 4.500 |
| article_part:5 | Dense kNN | 26 | 9 | 5 | 55.6% | 15.000 | 24.920 | 6.000 |
| article_part:5 | Lexical BM25-like | 26 | 9 | 5 | 55.6% | 17.400 | 27.659 | 6.000 |
| article_part:5 | Dense + structured | 26 | 9 | 4 | 44.4% | 4.500 | 6.708 | 3.000 |
| article_part:6 | Cohort median | 18 | 5 | 3 | 60.0% | 36.000 | 40.628 | 34.000 |
| article_part:6 | Structured kNN | 18 | 5 | 3 | 60.0% | 36.000 | 40.628 | 34.000 |
| article_part:6 | Dense kNN | 18 | 5 | 3 | 60.0% | 36.000 | 40.628 | 34.000 |
| article_part:6 | Lexical BM25-like | 18 | 5 | 3 | 60.0% | 36.000 | 40.628 | 34.000 |
| article_part:6 | Dense + structured | 18 | 5 | 3 | 60.0% | 36.000 | 40.628 | 34.000 |
| outcome:cancelled | Cohort median | 7 | 3 | 3 | 100.0% | 24.000 | 27.276 | 18.000 |
| outcome:cancelled | Structured kNN | 7 | 3 | 2 | 66.7% | 15.000 | 15.297 | 15.000 |
| outcome:cancelled | Dense kNN | 7 | 3 | 3 | 100.0% | 28.000 | 33.586 | 18.000 |
| outcome:cancelled | Lexical BM25-like | 7 | 3 | 3 | 100.0% | 30.000 | 36.824 | 18.000 |
| outcome:cancelled | Dense + structured | 7 | 3 | 2 | 66.7% | 15.000 | 15.297 | 15.000 |
| outcome:changed | Cohort median | 38 | 10 | 5 | 50.0% | 13.200 | 26.967 | 0.000 |
| outcome:changed | Structured kNN | 38 | 10 | 5 | 50.0% | 16.200 | 27.528 | 6.000 |
| outcome:changed | Dense kNN | 38 | 10 | 5 | 50.0% | 16.200 | 27.528 | 6.000 |
| outcome:changed | Lexical BM25-like | 38 | 10 | 5 | 50.0% | 17.400 | 27.659 | 6.000 |
| outcome:changed | Dense + structured | 38 | 10 | 5 | 50.0% | 15.600 | 27.495 | 6.000 |
| outcome:unchanged | Cohort median | 29 | 8 | 4 | 50.0% | 18.000 | 20.688 | 16.000 |
| outcome:unchanged | Structured kNN | 29 | 8 | 4 | 50.0% | 18.000 | 20.688 | 16.000 |
| outcome:unchanged | Dense kNN | 29 | 8 | 4 | 50.0% | 18.000 | 20.688 | 16.000 |
| outcome:unchanged | Lexical BM25-like | 29 | 8 | 4 | 50.0% | 18.000 | 20.688 | 16.000 |
| outcome:unchanged | Dense + structured | 29 | 8 | 4 | 50.0% | 18.000 | 20.688 | 16.000 |

### fine_rub, ошибки log1p(руб.)

| Срез | Метод | Запросов | Известных targets | Покрыто | Coverage | MAE | RMSE | Median AE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| article_part:1 | Cohort median | 1 | 0 | 0 | — | — | — | — |
| article_part:1 | Structured kNN | 1 | 0 | 0 | — | — | — | — |
| article_part:1 | Dense kNN | 1 | 0 | 0 | — | — | — | — |
| article_part:1 | Lexical BM25-like | 1 | 0 | 0 | — | — | — | — |
| article_part:1 | Dense + structured | 1 | 0 | 0 | — | — | — | — |
| article_part:2 | Cohort median | 8 | 0 | 0 | — | — | — | — |
| article_part:2 | Structured kNN | 8 | 0 | 0 | — | — | — | — |
| article_part:2 | Dense kNN | 8 | 0 | 0 | — | — | — | — |
| article_part:2 | Lexical BM25-like | 8 | 0 | 0 | — | — | — | — |
| article_part:2 | Dense + structured | 8 | 0 | 0 | — | — | — | — |
| article_part:3 | Cohort median | 21 | 1 | 1 | 100.0% | 0.0690 | 0.0690 | 0.0690 |
| article_part:3 | Structured kNN | 21 | 1 | 1 | 100.0% | 0.0690 | 0.0690 | 0.0690 |
| article_part:3 | Dense kNN | 21 | 1 | 1 | 100.0% | 0.0690 | 0.0690 | 0.0690 |
| article_part:3 | Lexical BM25-like | 21 | 1 | 1 | 100.0% | 0.0690 | 0.0690 | 0.0690 |
| article_part:3 | Dense + structured | 21 | 1 | 1 | 100.0% | 0.0690 | 0.0690 | 0.0690 |
| article_part:5 | Cohort median | 26 | 2 | 1 | 50.0% | 1.1939 | 1.1939 | 1.1939 |
| article_part:5 | Structured kNN | 26 | 2 | 0 | 0.0% | — | — | — |
| article_part:5 | Dense kNN | 26 | 2 | 1 | 50.0% | 1.1939 | 1.1939 | 1.1939 |
| article_part:5 | Lexical BM25-like | 26 | 2 | 1 | 50.0% | 1.1939 | 1.1939 | 1.1939 |
| article_part:5 | Dense + structured | 26 | 2 | 0 | 0.0% | — | — | — |
| article_part:6 | Cohort median | 18 | 3 | 0 | 0.0% | — | — | — |
| article_part:6 | Structured kNN | 18 | 3 | 0 | 0.0% | — | — | — |
| article_part:6 | Dense kNN | 18 | 3 | 0 | 0.0% | — | — | — |
| article_part:6 | Lexical BM25-like | 18 | 3 | 0 | 0.0% | — | — | — |
| article_part:6 | Dense + structured | 18 | 3 | 0 | 0.0% | — | — | — |
| outcome:cancelled | Cohort median | 7 | 1 | 1 | 100.0% | 1.1939 | 1.1939 | 1.1939 |
| outcome:cancelled | Structured kNN | 7 | 1 | 0 | 0.0% | — | — | — |
| outcome:cancelled | Dense kNN | 7 | 1 | 1 | 100.0% | 1.1939 | 1.1939 | 1.1939 |
| outcome:cancelled | Lexical BM25-like | 7 | 1 | 1 | 100.0% | 1.1939 | 1.1939 | 1.1939 |
| outcome:cancelled | Dense + structured | 7 | 1 | 0 | 0.0% | — | — | — |
| outcome:changed | Cohort median | 38 | 4 | 0 | 0.0% | — | — | — |
| outcome:changed | Structured kNN | 38 | 4 | 0 | 0.0% | — | — | — |
| outcome:changed | Dense kNN | 38 | 4 | 0 | 0.0% | — | — | — |
| outcome:changed | Lexical BM25-like | 38 | 4 | 0 | 0.0% | — | — | — |
| outcome:changed | Dense + structured | 38 | 4 | 0 | 0.0% | — | — | — |
| outcome:unchanged | Cohort median | 29 | 1 | 1 | 100.0% | 0.0690 | 0.0690 | 0.0690 |
| outcome:unchanged | Structured kNN | 29 | 1 | 1 | 100.0% | 0.0690 | 0.0690 | 0.0690 |
| outcome:unchanged | Dense kNN | 29 | 1 | 1 | 100.0% | 0.0690 | 0.0690 | 0.0690 |
| outcome:unchanged | Lexical BM25-like | 29 | 1 | 1 | 100.0% | 0.0690 | 0.0690 | 0.0690 |
| outcome:unchanged | Dense + structured | 29 | 1 | 1 | 100.0% | 0.0690 | 0.0690 | 0.0690 |

## Техническая проверка ranking

| Метод | Запросов с neighbors | Возвращено neighbors | Distance non-null | Distance null | Self exclusions нарушено | Cohort filter нарушено |
|---|---:|---:|---:|---:|---:|---:|
| Structured kNN | 69 | 232 | 232 | 0 | 0 | 0 |
| Dense kNN | 69 | 232 | 232 | 0 | 0 | 0 |
| Lexical BM25-like | 69 | 232 | 232 | 0 | 0 | 0 |
| Dense + structured | 69 | 232 | 232 | 0 | 0 | 0 |

- Structured без активных числовых признаков (unranked fallback): **0** запросов; у такого списка допустимы `distance=null`.
- Dense fallback: **{"empty_cohort": 5}**; dense+structured: **{"empty_cohort": 5}**.
- Отсутствующих embeddings внутри cohort: **0**.

## Сравнение и ограничения

- По prison_months MAE dense=19.750, structured=16.636, cohort median=17.500; это описание малой выборки, а не доказанное преимущество.
- По fine_rub MAE dense=0.6315, structured=0.0690, cohort median=0.6315; это описание малой выборки, а не доказанное преимущество.
- Выборка мала и нерепрезентативна: только подтверждённые полные пары одного пилота; внешней, временной или региональной проверки нет. Число известных targets существенно меньше числа запросов, поэтому RMSE неустойчив к отдельным делам.
- Пустые cohort и срезы с малым `n_known` означают отсутствие доказанного покрытия, а не нулевую ошибку. Прочерк в метрике означает отсутствие наблюдений.
- Одинаковые candidate-side и hard filters делают сравнение честным технически, но не устраняют ошибки извлечения признаков и targets из текстов.
- Dense+structured использует заранее заданный production вес 0.7; подбор веса по этой выборке не проводился. Доверительные интервалы и тесты значимости не дают надёжного вывода при таких counts.
- Локальный review pack с идентификаторами и excerpts предназначен только для ручной проверки lead. Он не публикуется; автоматическая оценка не заменяет эту проверку.
- **Production quality не доказано**: нужны больше независимых дел с известными targets, групповая внешняя/временная проверка и ручная оценка юридической уместности аналогов.
