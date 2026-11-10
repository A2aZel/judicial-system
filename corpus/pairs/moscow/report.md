# Подготовка полных пар

Отчёт содержит только агрегаты. Старые статусы подтверждения не использовались как доказательство.

- Кандидатов в индексе: **288**
- Событий в статусном файле: **385**
- Confirmed pairs: **75**
- Восстановлено официальных пар вне сводного индекса: **6**
- Missing sentence: **0**
- Missing appeal: **157**
- No compatible attachment: **53**
- Identity failures (включая integrity): **91**
- Duplicates: **0**

## По источнику и статусу

| Source | Status | Count |
| --- | --- | --- |
| expansion | confirmed_pair | 29 |
| expansion | identity_failed | 26 |
| expansion | missing_appeal | 1 |
| expansion | no_compatible_attachment | 9 |
| moscow_open_data | missing_appeal | 118 |
| moscow_open_data | no_compatible_attachment | 2 |
| mosgorsud | confirmed_pair | 45 |
| mosgorsud | identity_failed | 52 |
| mosgorsud | missing_appeal | 17 |
| mosgorsud | new_trial | 4 |
| mosgorsud | no_compatible_attachment | 30 |
| mosgorsud | procedural_act | 1 |
| processed | missing_appeal | 3 |
| recovery | confirmed_pair | 1 |
| recovery | identity_failed | 13 |
| recovery | missing_appeal | 5 |
| recovery | new_trial | 4 |
| recovery | no_compatible_attachment | 12 |
| sudact | missing_appeal | 13 |

## По году и статусу

| Year | Status | Count |
| --- | --- | --- |
| 2005 | missing_appeal | 1 |
| 2010 | missing_appeal | 5 |
| 2011 | missing_appeal | 2 |
| 2012 | missing_appeal | 2 |
| 2013 | missing_appeal | 1 |
| 2014 | missing_appeal | 1 |
| 2016 | missing_appeal | 21 |
| 2017 | missing_appeal | 24 |
| 2018 | confirmed_pair | 3 |
| 2018 | identity_failed | 1 |
| 2018 | missing_appeal | 17 |
| 2018 | no_compatible_attachment | 1 |
| 2019 | confirmed_pair | 11 |
| 2019 | identity_failed | 14 |
| 2019 | missing_appeal | 9 |
| 2019 | no_compatible_attachment | 7 |
| 2020 | confirmed_pair | 8 |
| 2020 | identity_failed | 10 |
| 2020 | missing_appeal | 5 |
| 2020 | no_compatible_attachment | 4 |
| 2021 | confirmed_pair | 1 |
| 2021 | identity_failed | 5 |
| 2021 | missing_appeal | 5 |
| 2021 | no_compatible_attachment | 4 |
| 2022 | confirmed_pair | 2 |
| 2022 | identity_failed | 2 |
| 2022 | missing_appeal | 22 |
| 2022 | no_compatible_attachment | 3 |
| 2023 | confirmed_pair | 1 |
| 2023 | identity_failed | 4 |
| 2023 | missing_appeal | 31 |
| 2023 | new_trial | 2 |
| 2023 | no_compatible_attachment | 12 |
| 2024 | confirmed_pair | 25 |
| 2024 | identity_failed | 21 |
| 2024 | missing_appeal | 6 |
| 2024 | new_trial | 3 |
| 2024 | no_compatible_attachment | 14 |
| 2024 | procedural_act | 1 |
| 2025 | confirmed_pair | 12 |
| 2025 | identity_failed | 27 |
| 2025 | missing_appeal | 4 |
| 2025 | new_trial | 3 |
| 2025 | no_compatible_attachment | 8 |
| 2026 | confirmed_pair | 12 |
| 2026 | identity_failed | 7 |
| 2026 | missing_appeal | 1 |

## По типу апелляционного акта

| Type | Count |
| --- | --- |
| missing_appeal | 131 |
| Уголовное дело (в апелляции) | 254 |

## Outcomes подтверждённых пар

| Outcome | Count |
| --- | --- |
| cancelled | 6 |
| changed | 41 |
| unchanged | 28 |

Процессуальные акты, направления на новое рассмотрение и сокращённые приговоры без мотивировки не входят в confirmed pairs. Этот pipeline не меняет processed dataset; включение выполняется отдельной пересборкой dataset.
