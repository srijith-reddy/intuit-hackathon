# Proxy-Block Dependence Diagnostic

Population: train, linked-feed complete cases (n=54,887 of 85,340; feed rate 64.3%).
Features: 13 proxies (bureau + bank-feed + behavioral). GraphicalLassoCV alpha=0.0001.

## Eigen / factor structure
- Variance shares: F1 31.9%, F2 13.7%, F3 8.1% (F1/F2 ratio 2.33)
- Factor-1 loadings (|.| sorted): credit_util +1.00, invoice_delinq +0.73, cash_p10 -0.71, debt_oblig +0.66, credit_band -0.60, payroll_reg -0.47, obs_revenue -0.36, decline_recency +0.34, overdrafts +0.28, inq_6mo +0.25, rev_volatility +0.20, inq_30d +0.16, rev_trend -0.03

## FDR-surviving edges (q=0.05), |partial corr| sorted
| edge | partial corr | p |
|---|---|---|
| cash_p10 — invoice_delinq | -0.778 | 0.0e+00 |
| credit_util — debt_oblig | +0.747 | 0.0e+00 |
| obs_revenue — payroll_reg | +0.504 | 0.0e+00 |
| credit_util — credit_band | -0.496 | 0.0e+00 |
| debt_oblig — credit_band | +0.352 | 0.0e+00 |
| rev_trend — rev_volatility | -0.322 | 0.0e+00 |
| obs_revenue — rev_trend | -0.313 | 0.0e+00 |
| credit_util — invoice_delinq | +0.307 | 0.0e+00 |
| debt_oblig — obs_revenue | -0.270 | 0.0e+00 |
| cash_p10 — payroll_reg | +0.244 | 0.0e+00 |
| credit_band — payroll_reg | +0.241 | 0.0e+00 |
| credit_util — rev_volatility | +0.236 | 0.0e+00 |
| debt_oblig — rev_volatility | -0.215 | 0.0e+00 |
| credit_band — rev_volatility | +0.195 | 0.0e+00 |
| rev_trend — payroll_reg | +0.182 | 0.0e+00 |
| credit_band — invoice_delinq | +0.168 | 0.0e+00 |
| debt_oblig — invoice_delinq | -0.147 | 5.6e-264 |
| credit_util — decline_recency | +0.135 | 9.9e-224 |
| credit_util — payroll_reg | +0.129 | 1.3e-202 |
| debt_oblig — cash_p10 | +0.124 | 2.6e-186 |
| credit_util — cash_p10 | -0.122 | 3.9e-180 |
| credit_util — inq_6mo | +0.106 | 5.7e-136 |
| obs_revenue — rev_volatility | -0.100 | 3.1e-123 |
| rev_volatility — payroll_reg | +0.094 | 1.7e-107 |
| credit_band — cash_p10 | +0.088 | 1.2e-95 |
| credit_util — rev_trend | -0.082 | 2.7e-83 |
| obs_revenue — cash_p10 | -0.081 | 5.5e-81 |
| inq_6mo — credit_band | -0.073 | 1.0e-65 |
| rev_volatility — invoice_delinq | +0.072 | 6.9e-65 |
| debt_oblig — rev_trend | +0.063 | 1.9e-49 |
| credit_band — rev_trend | -0.061 | 7.1e-46 |
| credit_util — inq_30d | +0.060 | 1.1e-45 |
| cash_p10 — overdrafts | -0.060 | 1.7e-45 |
| rev_volatility — cash_p10 | +0.057 | 3.1e-40 |
| decline_recency — inq_30d | +0.056 | 3.1e-39 |
| credit_util — overdrafts | +0.055 | 1.8e-38 |
| debt_oblig — decline_recency | -0.054 | 2.2e-36 |
| obs_revenue — invoice_delinq | +0.052 | 7.8e-34 |
| rev_trend — cash_p10 | -0.046 | 2.5e-27 |
| inq_6mo — debt_oblig | -0.044 | 4.4e-25 |
| inq_6mo — decline_recency | +0.044 | 2.0e-24 |
| credit_band — decline_recency | -0.041 | 9.4e-22 |
| overdrafts — invoice_delinq | +0.040 | 2.8e-21 |
| credit_util — obs_revenue | -0.040 | 5.1e-21 |
| payroll_reg — invoice_delinq | -0.036 | 3.0e-17 |
| credit_band — overdrafts | +0.034 | 7.6e-16 |
| debt_oblig — payroll_reg | -0.025 | 7.7e-09 |
| debt_oblig — inq_30d | -0.024 | 1.6e-08 |
| debt_oblig — overdrafts | -0.024 | 3.2e-08 |
| inq_6mo — invoice_delinq | -0.019 | 1.1e-05 |
| inq_6mo — inq_30d | +0.019 | 1.1e-05 |
| credit_band — obs_revenue | -0.018 | 2.8e-05 |
| inq_6mo — rev_volatility | -0.015 | 5.7e-04 |
| decline_recency — invoice_delinq | +0.009 | 2.6e-02 |

## Node degrees
credit_util:12, debt_oblig:12, credit_band:11, invoice_delinq:10, rev_volatility:9, cash_p10:9, obs_revenue:8, payroll_reg:8, inq_6mo:7, rev_trend:7, decline_recency:6, overdrafts:5, inq_30d:4

## Strong direct intervenable-pair edges (|pc| >= 0.20)
cash_p10—invoice_delinq (-0.778); credit_util—debt_oblig (+0.747); obs_revenue—payroll_reg (+0.504); credit_util—credit_band (-0.496); debt_oblig—credit_band (+0.352); rev_trend—rev_volatility (-0.322); obs_revenue—rev_trend (-0.313); credit_util—invoice_delinq (+0.307); debt_oblig—obs_revenue (-0.270); cash_p10—payroll_reg (+0.244); credit_band—payroll_reg (+0.241); credit_util—rev_volatility (+0.236); debt_oblig—rev_volatility (-0.215)

## VERDICT
PARTIAL — factor structure broadly supports a latent-health confounder, but specific intervenable pairs show strong direct edges (mediator risk): cash_p10—invoice_delinq (-0.78); credit_util—debt_oblig (+0.75); obs_revenue—payroll_reg (+0.50); credit_util—credit_band (-0.50); debt_oblig—credit_band (+0.35); rev_trend—rev_volatility (-0.32); obs_revenue—rev_trend (-0.31); credit_util—invoice_delinq (+0.31); debt_oblig—obs_revenue (-0.27); cash_p10—payroll_reg (+0.24); credit_band—payroll_reg (+0.24); credit_util—rev_volatility (+0.24); debt_oblig—rev_volatility (-0.22). For these features, prefer the heuristic / exclude the partner from conditioning sets.

## Cross-check vs shipped lambda_hat.csv
Features already on FALLBACK (heuristic) per the sanity gate: recent_inquiries_count_6mo, existing_debt_obligations, owner_personal_credit_band, multi_lender_inquiry_count_30d.
Audit below flags any 'use'-status feature contradicted by this diagnostic.
