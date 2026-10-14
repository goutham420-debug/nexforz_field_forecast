\## Feature Engineering \& Scaling (Day 7)



\### Feature Definitions \& Formulas

\- \*\*Baseline Features:\*\*

&#x20; - `temperature\_c`: Air temperature in polyhouse (°C).

&#x20; - `humidity\_pct`: Relative air humidity (%).

&#x20; - `co2\_ppm`: Carbon dioxide concentration (ppm).

\- \*\*Engineered Feature:\*\*

&#x20; - `temp\_humid\_interaction`: Interaction term capturing combined vapor/thermal stress on mushroom yield.

\- \*\*Scaling Strategy:\*\*

&#x20; - Applied `MinMaxScaler()` to map all feature columns to the range \[0, 1].

&#x20; - Serialized scaler object to `models/minmax\_scaler.joblib`.

&#x20; - Exported scaled dataset to `data/processed/features.parquet`.

