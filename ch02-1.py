from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt 

fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8, 
                10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7, 
                7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]


fish_data = [[l, w] for l, w in zip(fish_length, fish_weight)]
fish_target = [1] * 35 + [0] * 14

kn = KNeighborsClassifier()

# 0~4까지의 원소 
print(fish_data[0:5])

# 훈련 세트 입력값
train_input = fish_data[:35]

# 훈련 세트 타깃값 
train_target = fish_target[:35]

# 테스트 세트 입력값 
test_input = fish_data[35:]

# 테스트 세트 타깃값 
test_target = fish_target[35:]

# 훈련 세트로 모델 훈련 
kn.fit(train_input, train_target)
kn.score(test_input, test_target)

# 훈련 결과를 출력
print(kn.score(test_input, test_target))

# 넌파이 배열로 변환
input_arr = np.array(fish_data)
target_arr = np.array(fish_target)
print(input_arr)

# (샘플 수, 특성 수) 출력
print(input_arr.shape)

# 랜덤한 49개의 인덱스 생성
np.random.seed(42)
index = np.arange(49)
np.random.shuffle(index)

# 랜덤한 훈련 세트로 입력값과 타깃값 나누기
train_input = input_arr[index[:35]]
train_target = target_arr[index[:35]]

# 랜덤한 테스트 세트로 입력값과 타깃값 나누기
test_input = input_arr[index[35:]]
test_target = target_arr[index[35:]]

# 산점도 그리기
plt.scatter(train_input[:, 0], train_input[:, 1])
plt.scatter(test_input[:, 0], test_input[:, 1])
plt.xlabel('length')    
plt.ylabel('weight')
plt.show()

# 랜덤 데이터로 모델 훈련
kn.fit(train_input, train_target)

# 훈련 결과 출력
print(kn.score(test_input, test_target))

# 새로운 데이터 예측
print(kn.predict(test_input))