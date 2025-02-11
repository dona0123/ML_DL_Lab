import numpy as np 
import matplotlib.pyplot as plt 

# 이미지 데이터 불러오기
fruits = np.load('fruits_300.npy')
print(fruits.shape) # 샘플의 개수, 이미지 높이, 이미지 너비 

# 첫 번째 이미지의 첫 번째 행 출력
print(fruits[0, 0, :])

# 첫 번째 이미지 출력
plt.imshow(fruits[0], cmap='gray')
plt.show()

# 첫 번째 이미지의 색상을 반전하여 출력 
plt.imshow(fruits[0], cmap='gray_r')
plt.show()

# 바나나와 파인애플 이미지 출력 
fig, axs = plt.subplots(1, 2) # 1행 2열의 서브 플롯 생성
axs[0].imshow(fruits[100], cmap='gray_r')
axs[1].imshow(fruits[200], cmap='gray_r')
plt.show()

# 100 x 100 이미지를 펼쳐 10,000인 1차원 배열로 만들기 
apple = fruits[0:100].reshape(-1, 100*100)
pineapple = fruits[100:200].reshape(-1, 100*100)
banana = fruits[200:300].reshape(-1, 100*100)
print(apple.shape)

# 각 샘플의 픽셀 평균값 계산 
print(apple.mean(axis=1))

# 사과, 파인애플, 바나나에 대한 히스토그램 그리기 
plt.hist(np.mean(apple, axis=1), alpha=0.8) 
plt.hist(np.mean(pineapple, axis=1), alpha=0.8) 
plt.hist(np.mean(banana, axis=1), alpha=0.8) 
plt.legend(['apple', 'pineapple', 'banana'])
plt.show()

# 픽셀 10,000개에 대한 평균값을 막대그래프로 그리기 
fig, axs = plt.subplots(1, 3, figsize=(20, 5))
axs[0].bar(range(10000), np.mean(apple, axis=0))
axs[1].bar(range(10000), np.mean(pineapple, axis=0))
axs[2].bar(range(10000), np.mean(banana, axis=0))
plt.show()

# 픽셀 평균값을 100 x 100 크기로 바꿔 이미지처럼 출력 
apple_mean = np.mean(apple, axis=0).reshape(100, 100)
pineapple_mean = np.mean(pineapple, axis=0).reshape(100, 100)
banana_mean = np.mean(banana, axis=0).reshape(100, 100)
fig, axs = plt.subplots(1, 3, figsize=(20, 5))
axs[0].imshow(apple_mean, cmap='gray_r')
axs[1].imshow(pineapple_mean, cmap='gray_r')
axs[2].imshow(banana_mean, cmap='gray_r')
plt.show()

# 각 샘플의 오차 평균 구하기 
abs_diff = np.abs(fruits - apple_mean)
abs_mean = np.mean(abs_diff, axis=(1, 2)) # 샘플 1개당 평균 오차값 1개 
print(abs_mean.shape)

# apple_mean과 오차가 가장 작은 샘플 100개 고르기 

# 작은 순서대로 정렬 후 반환
apple_index = np.argsort(abs_mean)[:100] 

# 처음 100개를 선택해 10 x 10 격자로 이뤄진 그래프 출력 
fig, axs = plt.subplots(10, 10, figsize=(10, 10)) 
for i in range(10):
    for j in range(10): 
        axs[i, j].imshow(fruits[apple_index[i*10 + j]], cmap='gray_r')
        axs[i, j].axis('off')
plt.show()
