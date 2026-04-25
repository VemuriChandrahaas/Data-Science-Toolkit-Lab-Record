import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

data = pd.read_csv('temp_hum_play_data.csv')

le = LabelEncoder()
data['Play'] = le.fit_transform(data['Play'])

X = data[['Temperature', 'Humidity']]
y = data['Play']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = GaussianNB()
model.fit(X_train, y_train)

prediction = model.predict(X_test)
all_predictions = model.predict(X)

print("Accuracy:", accuracy_score(y_test, prediction))

print("\nClass Priors:", model.class_prior_)
print("\nMean (theta):\n", model.theta_)
print("\nVariance:\n", model.var_)

plt.figure(figsize=(8,6))

plt.scatter(
    X['Temperature'], X['Humidity'],
    c=y, cmap='coolwarm', marker='o', label='Actual'
)

plt.scatter(
    X['Temperature'], X['Humidity'],
    c=all_predictions, cmap='coolwarm',
    marker='x', s=100, label='Predicted'
)

plt.xlabel("Temperature")
plt.ylabel("Humidity")
plt.title("Actual vs Predicted (All Data)")
plt.legend()
plt.show()
