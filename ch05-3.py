import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# 데이터 프레임으로 변환
wine = pd.read_csv('https://bit.ly/wine_csv_data')

# 입력 데이터와 타깃 데이터로 나누기
data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()

# 훈련 세트와 테스트 세트로 나누기
train_input, test_input, train_target, test_target = train_test_split(data, target, test_size=0.2, random_state=42)

# 교차 검증 
rf = RandomForestClassifier(n_jobs=-1, random_state=42)
scores = cross_validate(rf, train_input, train_target, return_train_score=True, n_jobs=-1)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))

# 랜덤 포레스트 모델 훈련
rf.fit(train_input, train_target)

# 특성 중요도 출력 
print(rf.feature_importances_)

# OOB 점수(모델 평가 점수) 평균하여 출력 
rf = RandomForestClassifier(oob_score=True, n_jobs=-1, random_state=42)
rf.fit(train_input, train_target)
print(rf.oob_score_)

# ExtraTreesClassifier 모델의 교차 검증 점수 확인 
et = ExtraTreesClassifier(n_jobs=-1, random_state=42)
scores = cross_validate(et, train_input, train_target, return_train_score=True, n_jobs=-1)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))

# 특성 중요도 출력 
et.fit(train_input, train_target)
print(et.feature_importances_)

# GradientBoostingClassifier으로 와인 데이터셋의 교차 검증 점수 확인
gb = GradientBoostingClassifier(random_state=42)
scores = cross_validate(gb, train_input, train_target, return_train_score=True, n_jobs=-1)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))

# 결정 트리 개수를 500개로 5배 늘림 (그래도 과대적합 억제)
gb = GradientBoostingClassifier(n_estimators=500, learning_rate=0.2, random_state=42)
scores = cross_validate(gb, train_input, train_target, return_train_score=True, n_jobs=-1)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))

# 특성 중요도 출력 
gb.fit(train_input, train_target)
print(gb.feature_importances_)

# HistGradientBoostingClassifier(히스토그램 기반 그레이디언트 부스팅) 적용 
hgb = HistGradientBoostingClassifier(random_state=42)
scores = cross_validate(hgb, train_input, train_target, return_train_score=True)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))

# 모델을 훈련하고 훈련 세트에서 특성 중요도 계산 
hgb.fit(train_input, train_target)
result = permutation_importance(hgb, train_input, train_target, n_repeats=10, random_state=42, n_jobs=-1)
print(result.importances_mean)

# 테스트 세트에서 특성 중요도 계산 
result = permutation_importance(hgb, test_input, test_target, n_repeats=10, random_state=42, n_jobs=-1)
print(result.importances_mean)

# 테스트 세트에서의 성능 확인 
hgb.score(test_input, test_target)

# XGBClassifier(xgboost의 그레이디언트 부스팅 알고리즘) 사용해 와인 데이터 교차 검증 점수 확인 
xgb = XGBClassifier(tree_method='hist', random_state=42)
scores = cross_validate(xgb, train_input, train_target, return_train_score=True, verbose=1)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))

# LGBMClassifier(lightgbm의의 그레이디언트 부스팅 알고리즘) 
lgb = LGBMClassifier(random_state=42)
scores = cross_validate(lgb, train_input, train_target, return_train_score=True, n_jobs=-1)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))
