import numpy as np 
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.cluster import KMeans

# 이미지 데이터 불러오기
fruits = np.load('fruits_300.npy')

# (샘플 개수, 너비, 높이이) 3차원 배열을 (샘플 개수, 너비 x 높이) 2차원 배열로 변환
fruits_2d = fruits.reshape(-1, 100*100)

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

# PCA 모델 생성 및 훈련
pca = PCA(n_components=50) # 주성분 개수 50개로 설정
pca.fit(fruits_2d)

# 주성분이 담긴 배열의 크기 확인 
print(pca.components_.shape)

# 주성분을 그림으로 출력 (가장 분산이 큰 방향을 순서대로 나타냄)
draw_fruits(pca.components_.reshape(-1, 100, 100))

# 원본 데이터의 차원을 50으로 줄이기 
print(fruits_2d.shape)
fruits_pca = pca.transform(fruits_2d)
print(fruits_pca.shape)

# 10,000개의 특성을 복원 (다시 원본 차원으로)
fruits_inverse = pca.inverse_transform(fruits_pca)
print(fruits_inverse.shape)

# 복원한 이미지를 그림으로 출력
fruits_reconstruct = fruits_inverse.reshape(-1, 100, 100) 
for start in [0, 100, 200]: 
    draw_fruits(fruits_reconstruct[start:start+100])
    print('\n')

# 주성분이 설명하는 분산의 비율의 합 = 50개의 주성분으로 표현하고 있는 총 분산 비율 
print(np.sum(pca.explained_variance_ratio_)) # 0.9215 -> 원래 데이터의 정보(변화량) 중에서 92% 이상을 유지하면서 차원을 줄임 

# 설명된 분산의 비율을 그래프로 출력
plt.plot(pca.explained_variance_ratio_)
plt.show()

# 로지스틱 회귀 모델 생성 
lr = LogisticRegression()

# 타깃값 생성 
target = np.array([0]*100 + [1]*100 + [2]*100)

# 원본 데이터로 교차 검증 수행
scores = cross_validate(lr, fruits_2d, target)
print(np.mean(scores['test_score'])) # 교차 검증 점수 
print(np.mean(scores['fit_time'])) # 훈련 시간

# PCA 데이터로 교차 검증 수행
scores = cross_validate(lr, fruits_pca, target)
print(np.mean(scores['test_score'])) # 교차 검증 점수
print(np.mean(scores['fit_time'])) # 훈련 시간

# 분산의 50%를 유지하는 주성분 찾아 PCA 모델 만들기 
pca = PCA(n_components=0.5) 
pca.fit(fruits_2d)

# 주성분 개수 출력
print(pca.n_components_)

# 원본 데이터 변환 
fruits_pca = pca.transform(fruits_2d)
print(fruits_pca.shape)

# 교차 검증 수행
scores = cross_validate(lr, fruits_pca, target)
print(np.mean(scores['test_score'])) # 교차 검증 점수
print(np.mean(scores['fit_time'])) # 훈련 시간

# 차원 축소된 데이터를 사용해 k-평균 모델로 클러스터 찾기 
km = KMeans(n_clusters=3, random_state=42)
km.fit(fruits_2d)
print(np.unique(km.labels_, return_counts=True))

# KMeans가 찾은 레이블을 사용해 과일 이미지를 출력 
for label in range(0, 3):
    draw_fruits(fruits[km.labels_==label])
    print('\n')

# 클러스터별로 나누어 산점도 그리기 
for label in range(0, 3):
    data = fruits_pca[km.labels_==label]
    plt.scatter(data[:, 0], data[:, 1])
plt.legend(['apple', 'banana', 'pineapple'])
plt.show()
