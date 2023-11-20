import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
import matplotlib.pyplot as plt

# 예제 데이터셋 생성
data = {
    '날씨': ['맑음', '흐림', '비', '맑음', '흐림', '비', '맑음', '흐림', '비', '맑음'],
    '온도': ['저', '중', '중', '고', '저', '고', '저', '고', '중', '고'],
    '습도': ['저', '저', '고', '고', '저', '고', '저', '고', '저', '고'],
    '세탁량': ['적음', '적음', '적음', '많음', '적음', '많음', '많음', '많음', '적음', '많음'],
    '세탁기 사용': ['미사용', '미사용', '사용', '사용', '미사용', '사용', '미사용', '사용', '미사용', '사용']
}

df = pd.DataFrame(data)

# 범주형 데이터를 원-핫 인코딩을 통해 수치형으로 변환
# 범주형 데이터(Object 타입)가 가진 의미를 버리지 않고 함축된 의미를 유지핸 채 숫자형 데이터로 변경, 즉 0 또는 1로 변경
# 컬럼분류 ==> 온도_저, 온도_중, 온도_고, 습도_저, 습도_고, 세탁량_적음, 세탁량_많음, 세탁기사용_미사용, 세탁기사용_사용
df_encoded = pd.get_dummies(df)

# 특성과 타겟 분리
X = df_encoded.drop('세탁기 사용_사용', axis=1)
y = df_encoded['세탁기 사용_사용']

# 학습 및 테스트 데이터셋 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 결정 트리 모델 생성 및 학습
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# 테스트 데이터셋으로 성능 평가
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# 사용자 입력 받기
input_weather = input("날씨를 입력해주세요 (맑음/흐림/비): ")
input_temp = input("온도를 입력해주세요 (저/중/고): ")
input_humidity = input("습도를 입력해주세요 (저/고): ")
input_laundry = input("세탁량을 입력해주세요 (적음/많음): ")

# 입력받은 데이터를 데이터 프레임 형태로 변환
input_data = pd.DataFrame({
    '날씨': [input_weather],
    '온도': [input_temp],
    '습도': [input_humidity],
    '세탁량': [input_laundry]
})

# 원-핫 인코딩 적용
input_encoded = pd.get_dummies(input_data)
input_encoded = input_encoded.reindex(columns = X_train.columns, fill_value=0)

# 모델을 사용하여 예측
prediction = model.predict(input_encoded)

# 예측 결과 출력
prediction_result = '사용' if prediction[0] == 1 else '미사용'
print("세탁기 사용 여부 예측:", prediction_result)


#

# 결정 트리 시각화
plt.rc('font', family='Malgun Gothic')
plt.figure(figsize=(12, 8))
tree.plot_tree(model, feature_names=X.columns, class_names=['미사용', '사용'], filled=True)
# plt.show()
