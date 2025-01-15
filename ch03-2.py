import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

perch_length = np.array([8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0, 21.0,
       21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5, 22.5, 22.7,
       23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5, 27.3, 27.5, 27.5,
       27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0, 36.5, 36.0, 37.0, 37.0,
       39.0, 39.0, 39.0, 40.0, 40.0, 40.0, 40.0, 42.0, 43.0, 43.0, 43.5,
       44.0])
perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])

# 훈련 세트와 테스트 세트로 나누기
train_input, test_input, train_target, test_target = train_test_split(perch_length, perch_weight, random_state=42)

# 훈련 세트와 테스트 세트를 2차원 배열로 바꾸기
train_input = train_input.reshape(-1, 1)
test_input = test_input.reshape(-1, 1)

# k-최근접 이웃 회귀 객체 생성
knr = KNeighborsRegressor(n_neighbors=3)

# 모델 훈련
knr.fit(train_input, train_target)

# 길이가 50cm인 농어의 무게 예측
print(knr.predict([[50]])) # [1033.33333333] -> 실제와 다름

# 50cm 농어의 이웃을 구함
distances, indexes = knr.kneighbors([[50]])

# 훈련 세트의 산점도
plt.scatter(train_input, train_target)

# 훈련 세트 중 이웃 샘플만 다시 그림
plt.scatter(train_input[indexes], train_target[indexes], marker='D')

# 50cm 농어 데이터
plt.scatter(50, 1033, marker='^')
plt.xlabel('length')   
plt.ylabel('weight')
plt.show()

# 이웃 샘플의 타깃의 평균
print(np.mean(train_target[indexes])) # 1033.3333333333333 -> 실제와 다름

# 농어의 길이가 100cm일 때 무게 예측
print(knr.predict([[100]])) # [1033.33333333] -> 실제와 다름

# 100cm 농어의 이웃을 구함
distances, indexes = knr.kneighbors([[100]])

# 훈련 세트의 산점도
plt.scatter(train_input, train_target)

# 훈련 세트 중 이웃 샘플만 다시 그림
plt.scatter(train_input[indexes], train_target[indexes], marker='D')

# 100cm 농어 데이터
plt.scatter(100, 1033, marker='^')
plt.xlabel('length')  
plt.ylabel('weight')
plt.show()

# 선형 회귀 모델 
lr = LinearRegression()

# 선형 회귀 모델 훈련
lr.fit(train_input, train_target)

# 50cm 농어에 대한 예측
print(lr.predict([[50]])) 

# 기울기(a), 절편(b) 출력
print(lr.coef_, lr.intercept_) # [39.01714496] -709.0186449535477

# 훈련 세트의 산점도
plt.scatter(train_input, train_target)

# 15에서 50까지 1차 방정식 그래프
plt.plot([15, 50], [15*lr.coef_+lr.intercept_, 50*lr.coef_+lr.intercept_])

# 50cm 농어 데이터
plt.scatter(50, 1241.8, marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

# 훈련 세트와 테스트 세트에 대한 R^2 점수 
print(lr.score(train_input, train_target)) # 0.939
print(lr.score(test_input, test_target)) # 0.824

# 0g 이하 무게는 0으로 만들기
# 2차 방정식으로 변환
train_poly = np.column_stack((train_input ** 2, train_input))
test_poly = np.column_stack((test_input ** 2, test_input))

# 2차 방정식으로 변환된 데이터로 모델 훈련
lr = LinearRegression()
lr.fit(train_poly, train_target)

# 훈련된 모델 테스트 
print(lr.predict([[50**2, 50]])) # [1573.98423528]
print(lr.coef_, lr.intercept_) # [  1.01433211 -21.55792498] 116.05021078278276 -> 무게 = 1.01 * 길이^2 - 21.56 * 길이 + 116.05

# 구간별 직선을 그리기 위해 15에서 49까지 정수 배열 생성
point = np.arange(15, 50)

# 훈련 세트의 산점도
plt.scatter(train_input, train_target)

# x값 15에서 49까지 2차 방정식 그래프
plt.plot(point, 1.01*point**2 - 21.6*point +116.05)

# 50cm 농어 데이터
plt.scatter(50, 1574, marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

# 훈련 세트와 테스트 세트에 대한 R^2 점수
print(lr.score(train_poly, train_target)) # 0.970
print(lr.score(test_poly, test_target)) # 0.977