import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso

# csv 파일 -> 판다스 데이터프레임 -> 넘파이 배열
df = pd.read_csv('https://bit.ly/perch_csv_data')
perch_full = df.to_numpy()
print(perch_full)

# 타깃 데이터
perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])

# 훈련 세트와 테스트 세트로 나누기
train_input, test_input, train_target, test_target = train_test_split(perch_full, perch_weight, random_state=42)

# PolynomialFeatures 클래스로 특성 만들기
poly = PolynomialFeatures(include_bias=False)
poly.fit(train_input)
train_poly = poly.transform(train_input)
print(train_poly.shape)

# get_feature_names 메서드로 특성 이름 확인
print(poly.get_feature_names_out()) # ['x0', 'x1', 'x2', 'x0^2', 'x0 x1', 'x0 x2', 'x1^2', 'x1 x2', 'x2^2'] 출력 -> 차수 2로, 3개의 특성을 2개씩 조합하여 9개의 특성 생성

# LinearRegression 모델 훈련
lr = LinearRegression()
lr.fit(train_poly, train_target)
print(lr.score(train_poly, train_target)) # 0.9903183436982124 출력 -> 99% 정도의 정확도
print(lr.score(poly.transform(test_input), test_target)) # 0.9714559911594132 출력 -> 97% 정도의 정확도

# 5제곱까지 특성을 만들어 출력
poly = PolynomialFeatures(degree=5, include_bias=False)
poly.fit(train_input)
train_poly = poly.transform(train_input)
test_poly = poly.transform(test_input)
print(train_poly.shape) # (42, 55) 출력 -> 42개의 샘플, 55개의 특성 

# LinearRegression 모델 훈련
lr.fit(train_poly, train_target)
print(lr.score(train_poly, train_target)) # 0.9999999999991096 출력 -> 100% 정도의 정확도

# 테스트 세트 정확도 출력
print(lr.score(test_poly, test_target)) # -144.40579242335605 출력 -> 음수가 나와서 이상함

# 특성의 스케일을 조정, 표준점수로 변환
ss = StandardScaler()
ss.fit(train_poly)
train_scaled = ss.transform(train_poly)
test_scaled = ss.transform(test_poly)

# 릿지 회귀 모델 훈련
ridge = Ridge()
ridge.fit(train_scaled, train_target)
print(ridge.score(train_scaled, train_target)) # 0.9896101671037343 출력 -> 98% 정도의 정확도
print(ridge.score(test_scaled, test_target)) # 0.9790693977615398 출력 -> 97% 정도의 정확도

# 적절한 alpha 값(규제의 강도)을 찾기
# alpha 값에 따른 훈련 세트와 테스트 세트의 점수를 저장
train_score = []
test_score = []

alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alpha_list:
    # 릿지 모델 훈련
    ridge = Ridge(alpha=alpha)
    
    # 릿지 모델 훈련 
    ridge.fit(train_scaled, train_target)
    
    # 훈련 세트와 테스트 세트의 점수 저장
    train_score.append(ridge.score(train_scaled, train_target))
    test_score.append(ridge.score(test_scaled, test_target))

# alpha 값에 따른 훈련 세트와 테스트 세트의 점수 그래프로 출력
plt.plot(np.log10(alpha_list), train_score)
plt.plot(np.log10(alpha_list), test_score)
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.show() # 최적의 alpha 값은 -1, 즉 10^-1 = 0.1

# 최적의 alpha 값으로 모델 훈련
ridge = Ridge(alpha=0.1)
ridge.fit(train_scaled, train_target)
print(ridge.score(train_scaled, train_target)) # 0.9903815817570366 출력 -> 99% 정도의 정확도
print(ridge.score(test_scaled, test_target)) # 0.9827976465386927 출력 -> 98% 정도의 정확도 

# 라쏘 회귀 모델 훈련
lasso = Lasso()
lasso.fit(train_scaled, train_target)
print(lasso.score(train_scaled, train_target)) # 0.9897898972080961 출력 -> 98% 정도의 정확도
print(lasso.score(test_scaled, test_target)) # 0.9800593698421883 출력 -> 98% 정도의 정확도

# 적절한 alpha 값(규제의 강도)을 찾기
train_score = []
test_score = []
alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alpha_list:
    # 라쏘 모델 훈련
    lasso = Lasso(alpha=alpha, max_iter=10000) # 최대 반복 횟수를 10000으로 설정
    
    # 라쏘 모델 훈련
    lasso.fit(train_scaled, train_target)
    
    # 훈련 세트와 테스트 세트의 점수 저장
    train_score.append(lasso.score(train_scaled, train_target))
    test_score.append(lasso.score(test_scaled, test_target))

# alpha 값에 따른 훈련 세트와 테스트 세트의 점수 그래프로 출력
plt.plot(np.log10(alpha_list), train_score)
plt.plot(np.log10(alpha_list), test_score)
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.show() # 최적의 alpha 값은 1, 즉 10^1 = 10

# 최적의 alpha 값으로 모델 훈련
lasso = Lasso(alpha=10)
lasso.fit(train_scaled, train_target)
print(lasso.score(train_scaled, train_target)) # 0.9888067471131867 출력 -> 98% 정도의 정확도
print(lasso.score(test_scaled, test_target)) # 0.9824470598706695 출력 -> 98% 정도의 정확도

# 라쏘 모델의 계수 출력
print(np.sum(lasso.coef_ == 0)) # 40 출력 -> 55개의 특성 중 40개의 특성이 0으로 변환됨




