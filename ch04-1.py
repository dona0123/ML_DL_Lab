import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from scipy.special import expit
from scipy.special import softmax

# 데이터 프레임으로 변환
fish = pd.read_csv('https://bit.ly/fish_csv_data')

# 처음 5개 행 출력
print(fish.head())

# 생선의 종류 출력 
print(pd.unique(fish['Species']))

# 입력 데이터 생성
fish_input = fish[['Weight', 'Length', 'Diagonal', 'Height', 'Width']].to_numpy()

# 저장된 5개의 특성 중 처음 5행 출력
print(fish_input[:5])

# 타깃 데이터 생성 
fish_target = fish['Species'].to_numpy()

# 훈련 세트와 테스트 세트로 나누기
train_input, test_input, train_target, test_target = train_test_split(fish_input, fish_target, random_state=42)

# 훈련 세트와 데스트 세트를 표준화 전처리 
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

# 표준화 전처리된 훈련 세트로 k-최근접 이웃 모델 훈련
kn = KNeighborsClassifier(n_neighbors=3)
kn.fit(train_scaled, train_target)
print(kn.score(train_scaled, train_target))
print(kn.score(test_scaled, test_target))

# 정렬된 타깃값 출력
print(kn.classes_)

# 테스트 세트 처음 5개 샘플의 예측값 출력
print(kn.predict(test_scaled[:5]))

# 테스트 세트 처음 5개 샘플의 확률값 출력
proba = kn.predict_proba(test_scaled[:5])
print(np.round(proba, decimals=4)) # 소수점 4번째 자리까지 표기 

# 4번째 샘플의 최근접 이웃의 클래스 출력
distances, indexes = kn.kneighbors(test_scaled[3:4]) # 4번째 행(인덱스 3)
print(train_target[indexes])

# 그래프 그리기 
z = np.arange(-5, 5, 0.1) # -5부터 5까지 0.1 간격으로 배열 생성
phi = 1 / (1 + np.exp(-z))
plt.plot(z, phi) # z(x축)와 phi(y축)를 그래프로 그림
plt.xlabel('z')
plt.ylabel('phi')
plt.show()

# 도미와 방어 데이터만 선택
bream_smelt_indexes = (train_target == 'Bream') | (train_target == 'Smelt')
train_bream_smelt = train_scaled[bream_smelt_indexes]
target_bream_smelt = train_target[bream_smelt_indexes]

# 로지스틱 회귀 모델 훈련 
lr = LogisticRegression()
lr.fit(train_bream_smelt, target_bream_smelt)

# 훈련 세트 예측
print(lr.predict(train_bream_smelt[:5]))

# 훈련 세트 확률값 출력
print(lr.predict_proba(train_bream_smelt[:5])) # 음성 클래스(0), 양성 클래스(1) 순서로 출력
print(lr.classes_) # Bream, Smelt 순서로 출력

# 로직스틱 회귀가 학습한 계수 출력
print(lr.coef_, lr.intercept_) # z = -0.404 x (Weight) -0.575 x (Length) -0.662 x (Diagonal) -1.013 x (Height) -0.731 x (Width) -2.161

# z값 출력
decisions = lr.decision_function(train_bream_smelt[:5]) # 양성 클래스에 대한 z값 출력
print(decisions) 

# 시그모이드 함수로 확률값 출력
print(expit(decisions)) 

# 로지스틱 회귀로 다중 분류 수행
lr = LogisticRegression(C=20, max_iter=1000)
lr.fit(train_scaled, train_target)
print(lr.score(train_scaled, train_target))
print(lr.score(test_scaled, test_target))

# 테스트 세트 처음 5개 샘플의 예측값 출력
print(lr.predict(test_scaled[:5]))

# 테스트 세트 처음 5개 샘플의 확률값 출력
proba = lr.predict_proba(test_scaled[:5])
print(np.round(proba, decimals=3))

# 클래스 정보 출력
print(lr.classes_)

# 계수와 절편 출력
print(lr.coef_.shape, lr.intercept_.shape)

# z1~z7 출력
decisions = lr.decision_function(test_scaled[:5])
print(np.round(decisions, decimals=2))

# 소프트맥스 함수로 확률값 출력
proda = softmax(decisions, axis=1)
print(np.round(proba, decimals=3))