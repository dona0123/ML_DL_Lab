import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from scipy.stats import uniform, randint
from sklearn.model_selection import RandomizedSearchCV

# 데이터 프레임으로 변환
wine = pd.read_csv('https://bit.ly/wine_csv_data')

# 입력 데이터와 타깃 데이터로 나누기
data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()

# 훈련 세트와 테스트 세트로 나누기
train_input, test_input, train_target, test_target = train_test_split(data, target, test_size=0.2, random_state=42)

# 검증 세트로 나누기
sub_input, val_input, sub_target, val_target = train_test_split(train_input, train_target, test_size=0.2, random_state=42)

# 훈련 세트와 검증 세트의 크기 확인
print(sub_input.shape, val_input.shape)

# 결정 트리로 모델 훈련
dt = DecisionTreeClassifier(random_state=42)
dt.fit(sub_input, sub_target)
print(dt.score(sub_input, sub_target))
print(dt.score(val_input, val_target))

# 교차 검증
scores = cross_validate(dt, train_input, train_target)
print(scores)

# 교차 검증 점수의 평균
print(np.mean(scores['test_score']))

# StratifiedKFold를를 사용한 교차 검증 (앞서 수행한 교차 검증과 동일)
scores = cross_validate(dt, train_input, train_target, cv=StratifiedKFold())

# 훈련 세트를 섞은 후 10-폴드 교차 검증
splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_validate(dt, train_input, train_target, cv=splitter)
print(np.mean(scores['test_score']))

# min_impurity_decrease의 최적값 찾기 
params = {'min_impurity_decrease': [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]}

# 그리드 서치 객체 생성 및 훈련 
gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1)
gs.fit(train_input, train_target)

# 교차 검증 점수가 가장 높은 매개변수 조합의 모델 
dt = gs.best_estimator_
print(dt.score(train_input, train_target))

# 최적의 매개변수 출력 
print(gs.best_params_)

# 5번의 교차 검증으로 얻은 점수 출력 
print(gs.cv_results_['mean_test_score'])

# 최적의 매개변수 출력 (앞선 결과와 동일)
best_index = np.argmax(gs.cv_results_['mean_test_score'])
print(gs.cv_results_['params'][best_index])

# 복잡한 매개변수 조합 탐색 
params = {'min_impurity_decrease': np.arange(0.0001, 0.001, 0.0001), 'max_depth': range(5, 20, 1), 'min_samples_split': range(2, 100, 10)}

# 그리드 서치 
gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1)
gs.fit(train_input, train_target)

# 최상의 매개변수 조합 출력 
print(gs.best_params_)

# 최상의 교차 검증 점수 출력 
print(np.max(gs.cv_results_['mean_test_score']))

# 0에서 10 사이 범위를 갖는 randint 객체 생성 
rgen = randint(0, 10)

# 10개 숫자 샘플링 
print(rgen.rvs(10))

# 1000개를 샘플링하여 각 숫자의 개수 세어보기 
print(np.unique(rgen.rvs(1000), return_counts=True))

# 0~1 사이에서 10개의 실수 추출 
ugen = uniform(0, 1)
print(ugen.rvs(10))

# 탐색할 매개변수 범위 설정 
params = {'min_impurity_decrease': uniform(0.0001, 0.001), 'max_depth': randint(20, 50), 'min_samples_split': randint(2, 25), 'min_samples_leaf': randint(1, 25)}

# 100번 샘플링하여 교차 검증을 수행하고 최적의 매개변수 조합 찾기 
gs = RandomizedSearchCV(DecisionTreeClassifier(random_state=42), params, n_iter=100, n_jobs=-1, random_state=42)
gs.fit(train_input, train_target)

# 최적의 매개변수 조합 출력 
print(gs.best_params_)

# 최고의 교차 검증 점수 출력 
print(np.max(gs.cv_results_['mean_test_score']))

# 최적의 모델로 테스트 세트의 성능 확인 
dt = gs.best_estimator_
print(dt.score(test_input, test_target))
