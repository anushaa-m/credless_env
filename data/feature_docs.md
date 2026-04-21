# CredLess Dataset Documentation

## Source

- Dataset: `cd_updated.csv`
- Detected target column: `target`
- Inferred target semantics: `1_likely_good_outcome`
- Raw rows: 150,000
- Clean rows after duplicate/null filtering: 150,000
- Raw columns: 44
- Clean columns used by the model pipeline: 42

## Split Summary

- Train: 105,000 rows (70.0%)
- Validation: 22,500 rows (15.0%)
- Test: 22,500 rows (15.0%)

## Class Balance

- Full dataset: `{0: 0.5, 1: 0.5}`
- Train split: `{0: 0.5, 1: 0.5}`
- Validation split: `{0: 0.5, 1: 0.5}`
- Test split: `{0: 0.5, 1: 0.5}`

## Null Counts

All cleaned columns have zero null values after preprocessing.

## Column Inventory

| Column | Dtype | Nulls | Unique |
|---|---:|---:|---:|
| `revolvingutilizationofunsecuredlines` | `float64` | 0 | 124991 |
| `numberoftime30-59dayspastduenotworse` | `int64` | 0 | 1 |
| `numberoftimes90dayslate` | `int64` | 0 | 1 |
| `numberoftime60-89dayspastduenotworse` | `int64` | 0 | 1 |
| `numberrealestateloansorlines` | `int64` | 0 | 6 |
| `numberofdependents` | `float64` | 0 | 4 |
| `numberofopencreditlinesandloans` | `int64` | 0 | 21 |
| `age` | `int64` | 0 | 77 |
| `marital_status` | `int64` | 0 | 3 |
| `employment_type` | `int64` | 0 | 3 |
| `years_at_address` | `int64` | 0 | 20 |
| `months_employed` | `int64` | 0 | 239 |
| `monthlyincome` | `float64` | 0 | 10752 |
| `monthly_revenue` | `int64` | 0 | 123919 |
| `profit_margin` | `int64` | 0 | 30 |
| `business_age` | `int64` | 0 | 15 |
| `business_type_risk` | `float64` | 0 | 7 |
| `business_type_encoded` | `int64` | 0 | 7 |
| `debtratio` | `float64` | 0 | 106811 |
| `medical_debt` | `float64` | 0 | 5404 |
| `current_medical_condition` | `int64` | 0 | 1 |
| `emi_payment_ratio` | `float64` | 0 | 149976 |
| `rent_payment_regular` | `int64` | 0 | 1 |
| `utility_payment_ratio` | `float64` | 0 | 149975 |
| `fd_amount` | `int64` | 0 | 129624 |
| `gold_value_estimate` | `int64` | 0 | 105502 |
| `property_owned` | `int64` | 0 | 2 |
| `vehicle_owned` | `int64` | 0 | 2 |
| `emergency_savings_months` | `int64` | 0 | 12 |
| `bank_account_age_months` | `int64` | 0 | 174 |
| `avg_monthly_balance` | `int64` | 0 | 47090 |
| `negative_balance_days` | `int64` | 0 | 10 |
| `overdraft_count` | `int64` | 0 | 5 |
| `salary_credit_consistency` | `float64` | 0 | 149980 |
| `income_variability_score` | `float64` | 0 | 149991 |
| `monthly_upi_spend` | `int64` | 0 | 19890 |
| `active_txn_days` | `int64` | 0 | 25 |
| `failed_txn_ratio` | `float64` | 0 | 149935 |
| `late_night_txn_ratio` | `float64` | 0 | 149963 |
| `gov_scheme_enrollment` | `int64` | 0 | 4 |
| `location_risk_index` | `float64` | 0 | 91 |
| `target` | `int64` | 0 | 2 |
