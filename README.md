# Nexforz Mushroom Yield Forecast

## Problem Statement
Predict daily mushroom yield (kg) in a climate-controlled polyhouse using sensor telemetry including temperature (°C), relative humidity (%), and CO2 concentration (ppm).

## Project Structure
```text
nexforz-yield-forecast/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
│   └── smoke_test.py
├── models/
├── .gitignore
├── requirements.txt
└── README.md
```

## Environment Setup
Follow these steps to set up and run the environment locally:

1. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the polyhouse smoke-test script:
   ```bash
   python src/smoke_test.py
   ```