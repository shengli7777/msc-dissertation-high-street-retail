# MSc Dissertation: High Street Retail Change

This repository contains the code and selected outputs for an MSc Urban Spatial Science dissertation examining changes in High Street-related retail activity in Birmingham and Liverpool between 2016 and 2025.

## Research focus

The study examines:

- changes in the overall scale of retail activity;
- changes in retail functions and business diversity;
- spatial redistribution across 250 m grid cells;
- the evolution of commercial clusters identified using HDBSCAN;
- differences between Birmingham and Liverpool.

## Repository structure

- src/: Python scripts used for data processing, analysis and figure production.
- data/raw/: instructions for obtaining and placing the required input data.
- outputs/: selected analytical outputs, tables and figures.
- docs/: additional methodological documentation.
- requirements.txt: Python packages required to run the analysis.

## Main analytical workflow

The principal scripts should be run in the following order:

1. src/01_temporal_functional_analysis.py
2. src/02_grid_250m_spatial_analysis.py
3. src/03_hdbscan_cluster_analysis.py

Scripts 04-21 produce additional figures, maps and tables used in the dissertation.

## Data

The analysis uses OpenLocal commercial property records for Birmingham and Liverpool between 2016 and 2025. Local Authority District boundaries are obtained from the Office for National Statistics.

The original OpenLocal data are not included in this repository because the dataset was provided for dissertation analysis and may not be publicly redistributable. Required local filenames and locations are described in data/raw/README.md.

## Installation

Install the required Python packages with:

pip install -r requirements.txt

## Running the main scripts

python src/01_temporal_functional_analysis.py
python src/02_grid_250m_spatial_analysis.py
python src/03_hdbscan_cluster_analysis.py --input data/raw/openlocal_retail_property.parquet --output-dir outputs --filter retail
python src/03_hdbscan_cluster_analysis.py --input data/raw/openlocal_retail_property.parquet --output-dir outputs --filter retail_high_street

## Reproducibility

Most scripts use paths relative to the repository root. The required input files should be placed in data/raw/ before running the analysis. Selected derived outputs are provided in outputs/.

The outputs directory contains both intermediate analytical outputs and final dissertation figures. Figures used in the submitted dissertation are identified by their corresponding dissertation figure numbers or descriptive filenames.

## Methodological notes

The HDBSCAN workflow uses density-based clustering to identify commercial clusters from retail property records. Cluster matching, persistence, emergence, disappearance and centroid repositioning are documented in docs/HDBSCAN_methods_note.md.

The code and dissertation methodology should be read together. Parameter choices reported in the dissertation should match the corresponding scripts and documentation in this repository.

## Author

Sheng Li
MSc Urban Spatial Science
Centre for Advanced Spatial Analysis
University College London
