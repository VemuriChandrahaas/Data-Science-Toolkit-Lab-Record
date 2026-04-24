import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# NumPy - Creating Arrays
print("NumPy: Creating Arrays\n")

print(
    "1D Array:\n", np.array([1,2,3,4,5]), "\n\n",
    "2D Array:\n", np.array([[1,2,3],[4,5,6]]), "\n\n",
    "Zeros:\n", np.zeros((3,3)), "\n\n",
    "Ones:\n", np.ones((2,4)), "\n\n",
    "Identity:\n", np.eye(3), "\n\n",
    "Range:\n", np.arange(0,10,2)
)


# NumPy - Array Operations
print("\nNumPy: Array Operations\n")

a = np.array([1,2,3])
b = np.array([4,5,6])

print(
    "Array a:", a, "\n",
    "Array b:", b, "\n\n",
    "Addition:", a + b, "\n",
    "Multiplication:", a * b, "\n",
    "Scalar Multiplication:", a * 10, "\n\n",
    "Square Root:", np.sqrt(a), "\n",
    "Mean:", np.mean(a), "\n",
    "Dot Product:", np.dot(a,b)
)


# NumPy - Indexing and Slicing
print("\nNumPy: Indexing and Slicing\n")

matrix = np.array([[10,20,30],
                   [40,50,60],
                   [70,80,90]])

print(
    "Matrix:\n", matrix, "\n\n",
    "Element [1,2]:", matrix[1,2], "\n\n",
    "First Row:", matrix[0,:], "\n",
    "Second Column:", matrix[:,1], "\n\n",
    "Submatrix:\n", matrix[0:2,0:2]
)


# Pandas - Creating DataFrame
print("\nPandas: Creating DataFrame\n")

data = {
    'Name': ['Alice','Bob','Charlie'],
    'Age': [25,30,35],
    'City': ['New York','Paris','London']
}

df = pd.DataFrame(data)

print(df.head())
print("\n")
print(df.info())
print("\n")
print(df.describe())


# Pandas - Selection and Filtering
print("\nPandas: Selection and Filtering\n")

ages = df['Age']
print("Ages:\n", ages)

above_25 = df[df['Age'] > 25]
print("\nAbove 25:\n", above_25)

row_0 = df.iloc[0]
print("\nFirst Row:\n", row_0)

val = df.loc[0,'Name']
print("\nName at index 0:", val)


# Pandas - Data Cleaning
print("\nPandas: Data Cleaning\n")

data2 = {
    "Name": ["Alice","Bob","Charlie","David","Eva"],
    "Age": [23,np.nan,22,28,np.nan],
    "City": ["New York","London",np.nan,"Paris","Berlin"]
}

df2 = pd.DataFrame(data2)

print("Null values:\n", df2.isnull().sum())

print("\nDrop NA:\n", df2.dropna())

print("\nFill 0:\n", df2.fillna(0))

df2["Age"] = df2["Age"].fillna(df2["Age"].mean())
df2["City"] = df2["City"].fillna("Unknown")

print("\nCleaned Data:\n", df2)


# Matplotlib - Visualization
print("\nMatplotlib: Plots\n")

data3 = {
    'StudyHours':[1,2,3,4,5,6,7,8],
    'ExamScore':[35,40,50,55,65,70,78,85]
}

df3 = pd.DataFrame(data3)

plt.scatter(df3['StudyHours'], df3['ExamScore'])
plt.title("Scatter Plot")
plt.show()

plt.plot(df3['StudyHours'], df3['ExamScore'], marker='o')
plt.title("Line Plot")
plt.show()

plt.hist(df3['ExamScore'], bins=3)
plt.title("Histogram")
plt.show()

plt.bar(df3['StudyHours'], df3['ExamScore'])
plt.title("Bar Chart")
plt.show()


# Scikit-learn - Linear Regression
print("\nScikit-learn: Linear Regression\n")

data4 = {
    'Classes_Attended':[30,35,40,45,50,55,60,65,70,75],
    'Internal_Marks':[35,38,42,46,50,55,60,65,70,75]
}

df4 = pd.DataFrame(data4)

X = df4[['Classes_Attended']]
y = df4['Internal_Marks']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=7
)

model = LinearRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("MSE:", mean_squared_error(y_test, pred))
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

plt.scatter(X,y)
plt.plot(X, model.predict(X))
plt.title("Linear Regression")
plt.show()
