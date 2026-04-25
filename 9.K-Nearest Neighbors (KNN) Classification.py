import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv("mobile_price_category.csv")

X = data.drop("price_range", axis=1)
y = data["price_range"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

new_data = pd.DataFrame(
    [[842,0,2,0,1,7,0,0,0,1,2,0,0,7,0,0,1,1,0,0]],
    columns=X.columns
)

new_data = scaler.transform(new_data)

prediction = model.predict(new_data)[0]

labels = {
    0: "Low",
    1: "Medium",
    2: "High",
    3: "Very High"
}

print("Predicted Price Range:", labels[prediction])
