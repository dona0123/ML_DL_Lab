import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8, 
                10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7, 
                7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

# 2차원 리스트로 변환 (길이, 무게)
fish_data = np.column_stack((fish_length, fish_weight))

# 타깃 데이터 생성
fish_target = np.concatenate([np.ones(35), np.zeros(14)])

# 훈련 데이터와 테스트 데이터 나누기
train_input, test_input, train_target, test_target = train_test_split(fish_data, fish_target, random_state=42)

# 데이터 확인
print(train_input.shape, test_input.shape)
print(train_target.shape, test_target.shape) # (36,) (13,) -> 1차원 튜플

# 생선 비율에 맞게 다시 훈련 데이터와 테스트 데이터 나누기
train_input, test_input, train_target, test_target = train_test_split(fish_data, fish_target, stratify=fish_target, random_state=42)

# 모델 훈련하기
kn = KNeighborsClassifier()
kn.fit(train_input, train_target)
kn.score(test_input, test_target)

# 새로운 데이터 예측
print(kn.predict([[25, 150]])) # [0] -> 방어, 예측과 다름 

# 산점도 그리기 
plt.scatter(train_input[:, 0], train_input[:, 1])
plt.scatter(25, 150, marker='^')
plt.xlabel('Length')
plt.ylabel('Weight')
plt.show()

# 이웃까지의 거리, 이웃 샘플 인덱스 5개
distances, indexes = kn.kneighbors([[25, 150]])

# 가장 가까운 이웃 5개를 산점도에 표시
plt.scatter(train_input[:, 0], train_input[:, 1])
plt.scatter(25, 150, marker='^')
plt.scatter(train_input[indexes, 0], train_input[indexes, 1], marker='D')
plt.xlabel('Length')
plt.ylabel('Weight')
plt.show() 

# 가장 가까운 이웃 5개 출력
print(train_input[indexes])

# 데이터 전처리하여 산점도에 표시
plt.scatter(train_input[:, 0], train_input[:, 1])
plt.scatter(25, 150, marker='^')
plt.scatter(train_input[indexes, 0], train_input[indexes, 1], marker='D')
plt.xlim((0, 1000))
plt.xlabel('Length')
plt.ylabel('Weight')
plt.show()

# 평균 및 표준편차 계산
mean = np.mean(train_input, axis=0)
std = np.std(train_input, axis=0)
print(mean, std)

# 표준점수 계산 
train_scaled = (train_input - mean) / std

# 새로운 데이터도 표준점수로 변환하여 산점도에 표시
new = ([25, 150] - mean) / std
plt.scatter(train_scaled[:, 0], train_scaled[:, 1])
plt.scatter(new[0], new[1], marker='^')
plt.xlabel('Length')
plt.ylabel('Weight')
plt.show()

# 모델 훈련 및 점수 계산
kn.fit(train_scaled, train_target)
test_scaled = (test_input - mean) / std
print(kn.score(test_scaled, test_target))

# [1] -> 도미, 예측이 맞음
print(kn.predict([new])) 

# 예측 데이터(도미)를 기준으로 표준점수 변환 산점도 그리기 
distances, indexes = kn.kneighbors([new])
plt.scatter(train_scaled[:, 0], train_scaled[:, 1])
plt.scatter(new[0], new[1], marker='^')
plt.scatter(train_scaled[indexes, 0], train_scaled[indexes, 1], marker='D')
plt.xlabel('Length')
plt.ylabel('Weight')
plt.show()