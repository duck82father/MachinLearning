import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import mglearn

iris_dataset = load_iris()

# print("iris_dateset의 키:\n", iris_dataset.keys())
# print(iris_dataset['DESCR'][:193]+"\n...")
# print("타깃의 이름:", iris_dataset['target_names'])
# print("특성의 이름:\n", iris_dataset['feature_names'])
# print("data의 타입:", type(iris_dataset['data']))
# print("data의 크기:", iris_dataset['data'].shape)
# print("data의 처음 다섯 행:\n", iris_dataset['data'][:5])
# print("target의 타입:", type(iris_dataset['target']))
# print("target의 크기:", iris_dataset['target'].shape)
# print("target:\n", iris_dataset['target'])

X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

# print("X_train의 크기:", X_train.shape)
# print("y_train의 크기:", y_train.shape)
# print("X_test의 크기:", X_test.shape)
# print("y_test의 크기:", y_test.shape)

# X_train 데이터를 사용해서 데이터프레임을 만듭니다.
# 열의 이름은 iris_dataset.feature_names에 있는 문자열을 사용합니다.
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)

# 데이터프레임을 사용해 y_train에 따라 색으로 구분된 산점도 행렬을 만듭니다.
# pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15,15), marker='o', hist_kwds={'bins':20}, s=60, alpha=.8, cmap=mglearn.cm3)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

# 28
X_new = np.array([[5, 2.9, 1, 0.2]])
# print("X_new.shape:", X_new.shape)

# 29
prediction = knn.predict(X_new)
# print("예측:", prediction)
# print("예측한 타깃의 이름:", iris_dataset['target_names'][prediction])

# 30
y_pred = knn.predict(X_test)
# print("테스트 세트에 대한 예측값:\n", y_pred)

# 31
# print("테스트 세트의 정확도: {:.2f}".format(np.mean(y_pred==y_test)))
# print("테스트 세트의 정확도: {:.2f}".format(knn.score(X_test, y_test)))

# 33
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

print("테스트 세트의 정확도: {:.2f}".format(knn.score(X_test, y_test)))