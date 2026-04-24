import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

data = pd.read_csv("mall_customers.csv")

X = data[['Age','Annual Income (k$)','Spending Score (1-100)']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

model = KMeans(n_clusters=5, random_state=42)
y = model.fit_predict(X_pca)

plt.scatter(X_pca[:,0], X_pca[:,1], c=y)
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.title("KMeans Clustering")
plt.show()

new_data = pd.DataFrame([[25, 40, 60]], columns=X.columns)

new_scaled = scaler.transform(new_data)
new_pca = pca.transform(new_scaled)

cluster = model.predict(new_pca)[0]

print("Predicted Cluster:", "Group", cluster+1)
