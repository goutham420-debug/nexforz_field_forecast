\# Polyhouse Sensor Data Cleaning Log



\## 1. Column Definitions \& Agritech Meaning

\- `timestamp`: Observation date/time index at daily resolution.

\- `temperature\_c`: Microclimate temperature (°C). Expected oyster mushroom range is 10°C to 35°C.

\- `humidity\_pct`: Relative air humidity (%). Oyster mushrooms require 50% to 100% humidity.

\- `co2\_ppm`: Carbon dioxide concentration in parts per million. Typical range is 400 to 2000 ppm.

\- `yield\_kg`: Total harvested mushroom yield (target variable).



\## 2. Null Value Audit Before and After



| Column | Null Count (Before) | Null Count (After) | Strategy | Agritech Rationale |

|---|---|---|---|---|

| timestamp | 0 | 0 | None | Primary time index |

| temperature\_c | 0 | 0 | Forward-fill (limit=2) | Short sensor dropouts maintain prior microclimate state |

| humidity\_pct | 0 | 0 | Forward-fill (limit=2) | Short sensor dropouts maintain prior microclimate state |

| co2\_ppm | 0 | 0 | Forward-fill (limit=2) | Short sensor dropouts maintain prior microclimate state |

| yield\_kg | 0 | 0 | Drop missing | Target variable; never impute labels to prevent data leakage |



\## 3. Threshold Filtering \& Deduplication

\- Filtered ranges: `humidity\_pct` between 50 and 100, `temperature\_c` between 10 and 35, `co2\_ppm` between 400 and 2000.

\- Dropped duplicate records by timestamp keeping the last logged reading.

\- Saved processed dataset to `data/processed/02\_cleaned.parquet`.

