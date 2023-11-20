import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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

# 랜덤 포레스트 모델 생성 및 학습
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# 테스트 데이터셋으로 성능 평가
rf_y_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_y_pred)
print("랜덤 포레스트 정확도:", rf_accuracy)

# 사용자 입력 받기
input_weather = input("날씨를 입력해주세요 (맑음/흐림/비): ")
input_temp = input("온도를 입력해주세요 (저/중/고): ")
input_humidity = input("습도를 입력해주세요 (저/고): ")
input_laundry = input("세탁량을 입력해주세요 (적음/많음): ")

# 입력받은 데이터를 데이터 프레임 형태로 변환
input_data_rf = pd.DataFrame({
    '날씨': [input_weather],
    '온도': [input_temp],
    '습도': [input_humidity],
    '세탁량': [input_laundry]
})

# 원-핫 인코딩 적용
input_encoded_rf = pd.get_dummies(input_data_rf)
input_encoded_rf = input_encoded_rf.reindex(columns = X_train.columns, fill_value=0)

# 랜덤 포레스트 모델을 사용하여 예측
rf_prediction = rf_model.predict(input_encoded_rf)

# 예측 결과 출력
rf_prediction_result = '사용' if rf_prediction[0] == 1 else '미사용'
print("랜덤 포레스트를 사용한 세탁기 사용 여부 예측:", rf_prediction_result)
