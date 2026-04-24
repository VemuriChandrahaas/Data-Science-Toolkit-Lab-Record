import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

data = pd.read_csv("placement_salary.csv")

data['workex'] = data['workex'].map({'Yes':1, 'No':0})
data['specialisation'] = data['specialisation'].map({'Mkt&HR':0, 'Mkt&Fin':1})

X = data[['ssc_p','hsc_p','degree_p','etest_p','mba_p','workex','specialisation']]
y = data['salary']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

score = r2_score(y_test, y_pred)
print("Model R2 Score:", score)

new_candidate = pd.DataFrame([[85, 88, 82, 75, 80, 1, 1]], columns=X.columns)
predicted_salary = model.predict(new_candidate)
print("Predicted Salary:", predicted_salary[0])

plt.figure()
plt.scatter(X_test['mba_p'], y_test)
plt.xlabel("MBA Percentage")
plt.ylabel("Actual Salary")
plt.title("MBA % vs Salary")
plt.show()
