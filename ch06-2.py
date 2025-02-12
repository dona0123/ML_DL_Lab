import numpy as np 
from sklearn.cluster import KMeans 
import matplotlib.pyplot as plt 

# 이미지 데이터 불러오기
fruits = np.load('fruits_300.npy')

# (샘플 개수, 너비, 높이이) 3차원 배열을 (샘플 개수, 너비 x 높이) 2차원 배열로 변환
fruits_2d = fruits.reshape(-1, 100*100)

# KMeans 모델 생성 및 훈련 
km = KMeans(n_clusters=3, random_state=42) 
km.fit(fruits_2d)

# 훈련 결과 확인
print(km.labels_)

# 레이블 0, 1, 2로 모은 샘플의 개수 확인 
print(np.unique(km.labels_, return_counts=True))

# 각 클러스터가 어떤 이미지를 나타냈는지 그림으로 출력하는 함수 
def draw_fruits(arr, ratio=1):
    n = len(arr) # n은 샘플 개수 
    # 한 줄에 10개씩 이미지를 그림, 샘플 게수를 10으로 나누어 전체 행 개수를 계산 
    rows = int(np.ceil(n/10)) # 10으로 나누고 올림함 (행의 개수)

    # 행이 1개면 열의 개수는 샘플 개수, 그렇지 않으면 10개 
    cols = n if rows < 2 else 10 
    fig, axs = plt.subplots(rows, cols, figsize=(cols*ratio, rows*ratio), squeeze=False)
    for i in range(rows): 
        for j in range(cols): 
            if i*10 + j < n: # n개까지만 그림 
                axs[i, j].imshow(arr[i*10 + j], cmap='gray_r')
            axs[i, j].axis('off')
    plt.show()


# 함수를 사용하여 각 클러스터의 이미지들을 출력 
draw_fruits(fruits[km.labels_==0])
draw_fruits(fruits[km.labels_==1])
draw_fruits(fruits[km.labels_==2])

# 각 클러스터의 중심 출력 
draw_fruits(km.cluster_centers_.reshape(-1, 100, 100), ratio=3) 

# 샘플 1개의 클러스터 중심과의 거리 출력
print(km.transform(fruits_2d[100:101]))

# 가장 가까운 클러스터 중심을 예측 클래스로 출력 
print(km.predict(fruits_2d[100:101]))

# 샘플의 이미지 확인 
draw_fruits(fruits[100:101])

# 알고리즘이 반복한 횟수 출력 
print(km.n_iter_)

# 이너셔를 이용한 최적의 클러스터 개수 찾기

# 이너셔를 저장할 리스트 생성
inertia = []

# 클러스터 개수를 2에서 6까지 바꾸어가며 모델 생성 및 훈련
for k in range(2, 7):
    km = KMeans(n_clusters=k, n_init='auto', random_state=42) 
    km.fit(fruits_2d)
    inertia.append(km.inertia_)

# 클러스터 개수에 따른 이너셔 출력
plt.plot(range(2, 7), inertia)
plt.xlabel('k')
plt.ylabel('inertia')
plt.show()
