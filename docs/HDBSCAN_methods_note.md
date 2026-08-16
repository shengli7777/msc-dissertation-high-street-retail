# HDBSCAN cluster and moving-cluster outputs

This analysis clusters Birmingham and Liverpool retail activity in 2016 and 2025 using HDBSCAN.

Two definitions are provided:

- `retail`: all records where `category_group == "RETAIL"`.
- `retail_high_street`: strict subset where `category_subgroup == "RETAIL_HIGH_STREET"`.

Geometry is read from WKB point records. The raw coordinates are longitude/latitude, so the script converts them into local metre coordinates per city before HDBSCAN. Figures are plotted in longitude/latitude.

HDBSCAN is run separately for each city and year-window. The script then matches 2016 and 2025 clusters using centroid distance with a 1.2 km maximum match distance:

- `persistent_or_relocated`: a 2016 cluster matched to a 2025 cluster.
- `emerged`: a 2025 cluster without a 2016 match.
- `disappeared`: a 2016 cluster without a 2025 match.

Key strict high-street result:

- Birmingham: 24 clusters in 2016 and 20 clusters in 2025; 19 persistent/relocated, 1 emerged, 5 disappeared.
- Liverpool: 22 clusters in 2016 and 19 clusters in 2025; 19 persistent/relocated, 0 emerged, 3 disappeared.
- Matched strict high-street clusters shifted by a median of about 72 m, with the largest matched shift about 1.08 km.

Key broad retail-related result:

- Birmingham: 30 clusters in 2016 and 26 clusters in 2025; 26 persistent/relocated, 0 emerged, 4 disappeared.
- Liverpool: 18 clusters in 2016 and 18 clusters in 2025; 14 persistent/relocated, 4 emerged, 4 disappeared.
- Matched broad retail clusters shifted by a median of about 78 m, with the largest matched shift about 1.09 km.

Interpretive note: because the source data are fixed property locations, relocation is expressed mainly through cluster persistence, disappearance, emergence and changing cluster size, rather than large literal centroid movement.
