import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso
import matplotlib.pyplot as plt

np.random.seed(42)

X = np.random.randn(40, 15)

columns = [
    "House_Size",
    "Bedrooms",
    "Location_Rating",
    "Age_of_House",
    "Wall_Color",
    "Owner_Lucky_Number",
    "Street_Length",
    "Nearby_Trees",
    "Pet_Count",
    "Car_Model_Code",
    "Random_Noise_1",
    "Random_Noise_2",
    "Random_Noise_3",
    "Random_Noise_4",
    "Random_Noise_5"
]

df = pd.DataFrame(X, columns=columns)

df["House_Price"] = (
    15 * df["House_Size"] +
    8 * df["Bedrooms"] +
    12 * df["Location_Rating"] -
    5 * df["Age_of_House"] +
    np.random.randn(40) * 5
)

print("\n===== DATASET SAMPLE =====")
print(df.head())

X = df.drop("House_Price", axis=1)
y = df["House_Price"]

linear_model = LinearRegression()
linear_model.fit(X, y)

lasso_model = Lasso(alpha=1.0)
lasso_model.fit(X, y)

print("\n===== Linear Regression Coefficients =====")
print(pd.Series(np.round(linear_model.coef_, 2), index=X.columns))

print("\n===== Lasso Regression Coefficients =====")
print(pd.Series(np.round(lasso_model.coef_, 2), index=X.columns))

plt.figure(figsize=(14,6))

plt.bar(X.columns, linear_model.coef_, alpha=0.5, label='Linear')
plt.bar(X.columns, lasso_model.coef_, alpha=0.8, width=0.4, label='Lasso')

plt.xticks(rotation=90)
plt.axhline(0)
plt.legend()
plt.title("4 Real Features vs Noise Features")

plt.show()
