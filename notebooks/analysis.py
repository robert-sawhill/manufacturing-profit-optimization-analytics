import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Data
df = pd.read_csv('../data/manufacturing_data.csv')

# 2. Define Variables
X = df[['production_cost', 'labor_hours', 'machine_hours', 'raw_material_cost']]
y = df['profit']
X_const = sm.add_constant(X)

# 3. Multicollinearity Check (VIF)
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
print("--- Variance Inflation Factor (VIF) ---")
print(vif_data)

# 4. Fit Regression Model
model = sm.OLS(y, X_const).fit()
print("\n--- Regression Model Summary ---")
print(model.summary())

# 5. Residual Diagnostics
residuals = model.resid
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
sns.scatterplot(x=model.fittedvalues, y=residuals)
plt.axhline(0, color='red', linestyle='--')
plt.title('Residuals vs. Fitted Values')

plt.subplot(1, 2, 2)
sns.histplot(residuals, kde=True)
plt.title('Normality of Residuals')

plt.tight_layout()
plt.savefig('../images/regression_diagnostics.png')
