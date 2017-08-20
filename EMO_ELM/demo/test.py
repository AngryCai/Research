from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize
from EMO_ELM.classes.ELM_AE import ELM_AE
X, y = load_iris(return_X_y=True)
X = normalize(X)

elm_ae = ELM_AE(20, activation='sigmoid', sparse=True)
elm_ae.fit(X)
X_transform = elm_ae.predict(X)
X_train, X_test, y_train, y_test = train_test_split(X_transform, y, test_size=0.4, random_state=42, stratify=y)

from sklearn.neighbors import KNeighborsClassifier as KNN
knn_1 = KNN(3)
knn_1.fit(X_train, y_train)
y_1 = knn_1.predict(X_test)

acc = accuracy_score(y_test, y_1)
print(acc)