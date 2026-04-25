import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

data = load_breast_cancer()
X, y = data.data, data.target
features = data.feature_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

normal = LogisticRegression(penalty=None, max_iter=5000)
normal.fit(X_train, y_train)
normal_acc = accuracy_score(y_test, normal.predict(X_test))

ridge = LogisticRegression(penalty='l2', C=1.0, max_iter=5000)
ridge.fit(X_train, y_train)
ridge_acc = accuracy_score(y_test, ridge.predict(X_test))

print(f"Normal Logistic Accuracy: {normal_acc:.3f}")
print(f"Ridge (L2) Accuracy: {ridge_acc:.3f}")

print("\nNormal Logistic Coefficients:")
print(pd.Series(np.round(normal.coef_[0], 3), index=features))

print("\nRidge (L2) Coefficients:")
print(pd.Series(np.round(ridge.coef_[0], 3), index=features))

plt.figure(figsize=(10,5))
plt.plot(normal.coef_[0], label='Normal')
plt.plot(ridge.coef_[0], label='Ridge (L2)', marker='o')
plt.axhline(0)
plt.legend()
plt.title("Coefficient Comparison")
plt.show()
