\# Task 5: Baseline Linear Regression Diagnostics \& Residual Analysis



\## 1. Test Performance Metrics

\- \*\*Mean Absolute Error (MAE):\*\* Measures average deviation on unseen test batches (\~kg/day)\[cite: 11, 12].

\- \*\*Root Mean Squared Error (RMSE):\*\* Captures variance penalty for larger prediction errors\[cite: 12].

\- \*\*Coefficient of Determination ($R^2$):\*\* Quantifies variance explained by linear environmental combinations\[cite: 11, 12].



\## 2. Coefficient Interpretation (Agritech Context)

\- \*\*Temperature (`temperature\_c\_scaled`):\*\* Positive coefficient indicates increased metabolic activity and fruiting acceleration within bounded operational ranges\[cite: 11].

\- \*\*Humidity (`humidity\_pct\_scaled`):\*\* Positive coefficient reflects prevention of basidiocarp desiccation and abortion of young pinheads\[cite: 11].

\- \*\*CO2 (`co2\_ppm\_scaled`):\*\* Negative coefficient aligns with agronomic expectations—excessive $\\text{CO}\_2$ (>900–1000 ppm) hampers cap expansion, prompting stem elongation and lowering net harvest quality\[cite: 11].

\- \*\*Interaction (`temp\_humid\_interaction\_scaled`):\*\* Captures simultaneous heat and vapor balance, demonstrating joint microclimatic impact beyond single-sensor assumptions.



\## 3. Residual Diagnostics (`reports/figures/residuals\_linear.png`)

\- \*\*Residuals vs. Predicted Yield:\*\* Residuals fluctuate symmetrically around the zero reference line ($e = y - \\hat{y}$), indicating that the baseline model is not severely biased toward systematic under- or over-prediction across mean harvest yields\[cite: 15, 17, 18].

\- \*\*Residuals vs. Scaled Humidity:\*\* Minimal funneling or heteroscedasticity across middle humidity ranges\[cite: 15, 16]. Mild curvature at boundary conditions suggests subtle non-linear thermal/moisture coupling\[cite: 15, 16].



\## 4. Modeling Recommendation

\- \*\*Baseline Verdict:\*\* The linear regression model establishes an interpretable benchmark for farm managers\[cite: 11, 12].

\- \*\*Next Step:\*\* Proceed to non-linear tree-based ensembles (e.g., Random Forest) in Day 11 to test whether decision splits better capture microclimate threshold triggers without overfitting\[cite: 11, 15, 18].

