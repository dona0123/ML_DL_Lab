import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt 
from sklearn.tree import plot_tree 

# 데이터 프레임으로 변환
wine = pd.read_csv('https://bit.ly/wine_csv_data')

# 5개의 데이터 확인
print(wine.head())

# 데이터 정보 확인
wine.info()

# 데이터 통계 확인
print(wine.describe())

# 입력 데이터와 타깃 데이터로 나누기
data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()

# 훈련 세트와 테스트 세트로 나누기
train_input, test_input, train_target, test_target = train_test_split(data, target, test_size=0.2, random_state=42)

# 훈련 세트와 테스트 세트의 크기 확인
print(train_input.shape, test_input.shape)

# 훈련 세트와 테스트 세트를 표준화 전처리
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

# 로지스틱 회귀로 모델 훈련
lr = LogisticRegression()
lr.fit(train_scaled, train_target)
print(lr.score(train_scaled, train_target))
print(lr.score(test_scaled, test_target))

# 계수와 절편 확인
print(lr.coef_, lr.intercept_)

# 결정 트리로 모델 훈련
dt = DecisionTreeClassifier(random_state=42)
dt.fit(train_scaled, train_target)
print(dt.score(train_scaled, train_target))
print(dt.score(test_scaled, test_target))

# 트리 그래프로 표현
plt.figure(figsize=(10, 7))
plot_tree(dt, max_depth=1, filled=True, feature_names=['alcohol', 'sugar', 'pH'])
plt.show()

# 트리의 깊이를 제한하여 모델 훈련
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(train_scaled, train_target)
print(dt.score(train_scaled, train_target))
print(dt.score(test_scaled, test_target))

# 트리 그래프로 표현
plt.figure(figsize=(20, 15))
plot_tree(dt, filled=True, feature_names=['alcohol', 'sugar', 'pH'])
plt.show()

# 전처리하기 전의 훈련 세트와 테스트 세트로 모델 훈련
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(train_input, train_target)  
print(dt.score(train_input, train_target))
print(dt.score(test_input, test_target))

# 트리 그래프로 표현
plt.figure(figsize=(20, 15))
plot_tree(dt, filled=True, feature_names=['alcohol', 'sugar', 'pH'])
plt.show()

# 특성 중요도 확인
print(dt.feature_importances_)
