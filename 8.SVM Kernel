from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

linear = SVC(kernel='linear')
poly = SVC(kernel='poly')
rbf = SVC(kernel='rbf')

linear.fit(X_train, y_train)
poly.fit(X_train, y_train)
rbf.fit(X_train, y_train)

acc_linear = accuracy_score(y_test, linear.predict(X_test))
acc_poly = accuracy_score(y_test, poly.predict(X_test))
acc_rbf = accuracy_score(y_test, rbf.predict(X_test))

print("Linear:", acc_linear)
print("Poly :", acc_poly)
print("RBF :", acc_rbf)

plt.bar(["Linear", "Poly", "RBF"], [acc_linear, acc_poly, acc_rbf])
plt.ylabel("Accuracy")
plt.title("SVM Kernel Comparison")
plt.show()
