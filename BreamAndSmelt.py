import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# 도미 데이터 준비하기
bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0,
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0,
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]
bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0,
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0,
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]

# 방어 데이터 준비하기
smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

# 그래프 생성하기
plt.scatter(bream_length, bream_weight, color='blue', label='Bream')
plt.scatter(smelt_length, smelt_weight, color='red', label='Smelt')
plt.xlabel('Length')
plt.ylabel('Weight')
plt.legend()
plt.show()

# 두 데이터 합치기
length = bream_length + smelt_length
weight = bream_weight + smelt_weight

# 데이터 정렬하기 (2차원 리스트)
fish_data = [[l, w] for l, w in zip(length, weight)]
fish_target = [1] * 35 + [0] * 14  # 도미는 1, 방어는 0

# 훈련 데이터와 테스트 데이터 나누기
X_train, X_test, y_train, y_test = train_test_split(fish_data, fish_target, test_size=0.2, random_state=42)

# 모델 훈련하기
kn = KNeighborsClassifier()
kn.fit(X_train, y_train)

# 훈련 결과 확인하기
print("훈련 데이터 점수:", kn.score(X_train, y_train))
print("테스트 데이터 점수:", kn.score(X_test, y_test))

# 새로운 데이터 예측
print("새 데이터 예측 결과:", kn.predict([[30, 600]]))

# n_neighbors=49 모델
kn49 = KNeighborsClassifier(n_neighbors=49)
kn49.fit(X_train, y_train)
print("n_neighbors=49 모델 점수:", kn49.score(X_test, y_test))
