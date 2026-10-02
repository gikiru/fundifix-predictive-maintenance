# Data Dictionary

## Primary dataset: wpdx_enhanced.csv

21,953 records, 54 fields. Source: WPdx-Plus Kenya, OCHA Humanitarian Data Exchange.

### Modeling features (retained)

| Field | Type | Missing | Description |
|---|---|---|---|
| lat_deg | float | 0% | Latitude |
| lon_deg | float | 0% | Longitude |
| water_source_clean | string | 0.6% | Cleaned water source type (e.g., Borehole, Spring) |
| water_tech_clean | string | 27.0% | Cleaned water technology (e.g., Hand Pump, Mechanized Pump) |
| water_source_category | string | 0.6% | Broader source category |
| water_tech_category | string | 30.1% | Broader technology category |
| install_year | float | 53.7% | Year of installation (missingness retained as a feature) |
| management_clean | string | 54.2% | Cleaned management entity type |
| facility_type | string | 0% | Facility type (e.g., Water Point, Improved Water Point) |
| local_population | float | 4.0% | Local population served |
| assigned_population | float | 4.0% | Population assigned to this water point |
| usage_cap | float | 4.8% | Usage capacity metric |
| criticality | float | 4.4% | Criticality score |
| pressure | float | 8.5% | Demand pressure metric |
| distance_to_primary | float | 0% | Distance to nearest primary road (metres) |
| distance_to_secondary | float | 0% | Distance to nearest secondary road (metres) |
| distance_to_tertiary | float | 0% | Distance to nearest tertiary road (metres) |
| distance_to_city | float | 0.2% | Distance to nearest city (metres) |
| distance_to_town | float | 0% | Distance to nearest town (metres) |
| is_urban | bool | 0% | Urban/rural classification |
| days_since_report | int | 0% | Days since the status was last reported |
| staleness | float | 0% | Staleness score (derived from days_since_report) |
| clean_adm1 | string | 0% | County |
| clean_adm2 | string | 0% | Sub-county |
| clean_adm3 | string | 0% | Ward |

### Target variable

| Field | Type | Missing | Description |
|---|---|---|---|
| status_clean | string | 0% | Six-category functional status. Collapsed to three classes for modeling: **Functional** (includes "Functional" + "Functional, not in use"), **Needs Repair**, **Non-Functional** (includes "Non-Functional" + "Non-Functional, dry season" + "Abandoned/Decommissioned"). |

Class distribution: Functional 30.6%, Needs Repair 6.7%, Non-Functional 62.7%.

### Excluded fields (not used in modeling)

| Field | Reason for exclusion |
|---|---|
| status_id | Measures a different condition than status_clean; incorrect target |
| installer | 82.5% missing |
| pay_clean | 83.6% missing |
| rehab_year | 100% missing |
| rehabilitator | 100% missing |
| fecal_coliform_presence | 100% missing |
| fecal_coliform_value | 100% missing |
| subjective_quality | 53.4% missing, subjective |
| notes | 84.8% missing, free-text |
| predicted_status_0y/2y, prediction_yes/no_0y/2y, predicted_category | All 100% missing (empty pre-computed columns) |
| clean_adm4, scheme_id | 100% missing |

## Secondary dataset: adm_analysis.csv

4,671 records (ward-level), 22 fields. Used for population-impact weighting and geographic equity auditing, not for model training.

| Field | Type | Description |
|---|---|---|
| NAME_1 / NAME_2 / NAME_3 | string | County / Sub-county / Ward |
| total_pop / urban_pop / rural_pop | float | Population figures |
| rural_pop_with_basic_access | float | Rural population with basic water access |
| rural_pop_without_basic_access | float | Rural population without basic water access |
| non_func_waterpoints | int | Non-functional water points in this ward |
| func_waterpoints | int | Functional water points in this ward |
| staleness_score | float | Average data staleness for this ward |
| data_quality_score | float | Data quality score for this ward |
