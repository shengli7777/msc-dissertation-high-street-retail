# Remade dissertation progress figures

These figures replace the earlier standalone drafts with a consistent Python/matplotlib style and a clearer evidential structure.

- `fig01_retail_scale_floor_area`: combines unit records, total floor area, median/IQR floor area and valid floor-area data coverage.
- `fig02_occupation_vacancy_data_quality`: separates occupation-status composition from vacancy rate among known cases.
- `fig03_retail_mix_and_shannon`: combines retail-subgroup composition, Shannon diversity and 2016-2025 subgroup share change.
- `fig04_floor_area_distribution_2016_2025`: shows the floor-area distribution shift using log-scaled violins.
- `fig05_spatial_floor_area_change`: maps high-street retail floor-area change with LAD boundaries.

Important interpretation notes:

- Floor-area values are missing for 2017 in the provided source, so floor-area totals and medians should be interpreted from years with valid floor-area coverage.
- Unknown occupation status varies substantially over time, so vacancy-rate claims should use the known-status denominator and mention the changing unknown-status share as a data-coverage limitation.
