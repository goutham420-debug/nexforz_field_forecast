\# Task 3: Exploratory Data Analysis \& Yield Correlation Report



\## 1. Dataset Overview \& Data Quality

\- \*\*Total Observations:\*\* 365 daily records (2024-01-01 to 2024-12-31).

\- \*\*Quality Audit:\*\* Zero missing values, zero out-of-bounds telemetry records, and zero duplicate timestamps post-cleaning.

\- \*\*Yield Distribution:\*\* Mean yield is 17.14 kg (std: 0.68 kg, min: 15.31 kg, max: 18.85 kg). The coefficient of variation (CV) is 0.04, reflecting stable production across the annual cycle.



\## 2. Sensor Summary Statistics

\- \*\*Temperature (`temperature\_c`):\*\* Mean = 21.99°C (std: 1.41°C, range: 18.15°C – 26.37°C, CV: 0.06).

\- \*\*Humidity (`humidity\_pct`):\*\* Mean = 86.74% (std: 3.07%, range: 78.10% – 94.80%, CV: 0.04).

\- \*\*CO2 (`co2\_ppm`):\*\* Mean = 901.16 ppm (std: 78.27 ppm, range: 608.00 – 1154.00 ppm, CV: 0.09).



\## 3. Sensor Correlations \& Biological Agritech Insights

\- \*\*Temperature vs. Yield:\*\* Demonstrates a moderate positive correlation. Within the optimal 18°C–25°C window for oyster mushrooms, warmer temperatures accelerate mycelial metabolism and fruiting flush cycles.

\- \*\*Humidity vs. Yield:\*\* Positive relationship. Maintaining relative humidity above 80% prevents basidiocarp desiccation and abortion of young pinheads.

\- \*\*CO2 vs. Yield:\*\* Inverted relationship. Higher $\\text{CO}\_2$ concentration (>950 ppm) suppresses pileus (cap) expansion and causes unwanted stipe (stem) elongation, directly reducing marketable yield weight.

\- \*\*Operational Takeaway:\*\* Automated polyhouse ventilation should trigger when $\\text{CO}\_2$ levels rise, while misting systems must sustain relative humidity near 85%–90%.



\## 4. Generated Publication Figures (`reports/figures/`)

\- `corr\_heatmap.png`: Pearson correlation matrix across telemetry features.

\- `scatter\_yield.png`: Tri-panel scatter plots mapping Humidity, Temperature, and CO2 against Yield.

\- `scatter\_humidity\_pct.png`, `scatter\_temperature\_c.png`, `scatter\_co2\_ppm.png`: Dedicated single-variable telemetry plots.

