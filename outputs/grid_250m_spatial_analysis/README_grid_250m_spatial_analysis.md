# 250 m grid-based spatial analysis

Spatial unit: 250 m x 250 m grid cells in British National Grid (EPSG:27700).

Input records: retail records for Birmingham and Liverpool in 2016 and 2025.

Per-cell metrics:

- `business_count`: number of retail records in the grid cell.
- `vacancy_rate`: vacant / (occupied + vacant), excluding unknown occupation status from the denominator.
- `business_diversity`: Shannon diversity index across retail subgroups.
- `rateable_value_total`: sum of rateable value within the cell.
- `rateable_value_median`: median rateable value within the cell.

Change metrics compare 2025 against 2016. Cell status is defined from business-count change:

- `emerged`: no 2016 retail records, positive 2025 records.
- `disappeared`: positive 2016 records, no 2025 records.
- `intensified`: positive in both years, count increased.
- `declined`: positive in both years, count decreased.
- `persistent`: positive in both years, count unchanged.

Interpretive use:

- Decline is indicated by declining/disappeared cells, rising vacancy rate, and falling rateable value.
- Relocation is indicated by simultaneous disappeared/declined cells in one area and emerged/intensified cells elsewhere.
- Adaptation is indicated by stable or rising activity with increased business diversity or changed value structure.
