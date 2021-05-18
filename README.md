# **Dataset :신용카드 사용자 데이터**

### **데이터 선정 이유**

코로나가 장기화되며 은행들이 손해를 최소화하기위해 신용등급 평가 기준을 재조정하며 신용카드 발급 기준을 향상시키고있다. 
발급 기준의 정확도를 향상시키기 위해 신용카드 신청자가 제출찬 개인 신상정보를 분석하고 주어진 데이터와 대금 연체 정도의 관계성을 파악해 더 정확한 평가 기준을 완성시키고자한다.  

### 데이터

신용카드 사용자들의 개인 신상정보를 사용해 분석합니다.  

### 분석목표

신용카드 사용자들의 개인 신상정보를 사용해 사용자의 대금 연체 정도를 예측할 수 있는 인공지능 알고리즘을 개발하자.  


### 분석 과정

#### **1.EDA**

1. ProfileReport를 통한 프로파일링 진행(pandas_profiling)

2. 주요 feature들을 시각화하여 데이터 분석

   -시각화에는 seaborn,matplot 사용
   
####**2.Feature Engineering**

1. 데이터 분석 결과 바탕으로 이상치 처리(Handling Outliers)

2. 범주형 변수가 주를 이루는 데이터이므로 TargetEncoder 사용하여 encoding

3. SimpleImputer  사용하여 결측치 처리

***train:test로 85:15로 데이터를 나누고 나눠진 train data에 대하여 train:val로 8:2로 나눠서 진행*** 



####**3.모델 및 성능 측정 기준 선택**

1. target이 0,1,2인 다중분류모델
2. 특정 값에 해당하는 데이터가 많이 분포되있는 imbalanced data이므로  F1-Score를 이용하여 성능 측정
3. RandomForest Classifier와 Light GBM 2가지 모델을 이용하여 트레이닝

####**4.모델 하이퍼파라미터 튜닝**

1. RandomizedSearchCV와 GridSearchCV 이용하여 최적 모델 탐색.

2. 파이프라이닝을 통해 TargetEncoder와 SimpleImputer도 한번에 진행

3. strategy(simpleimputer),smoothing(targetencoder),max_depth,max_features,n_estimators 등 조정

4. F1_score값과  AUC score 값을 평가하여 최종 모델 선정

   

####***최종 선정 모델*** 

```python
encoder = TargetEncoder(smoothing= 1.0)
X_train_encoded = encoder.fit_transform(X_train, y_train) # 학습데이터
X_val_encoded = encoder.transform(X_val, y_val) # 검증데이터

transformer = SimpleImputer(strategy='mean')
X_train_Imputed= transformer.fit_transform(X_train_encoded )
X_val_Imputed= transformer.fit_transform(X_val_encoded )

model = RandomForestClassifier(max_depth = 20,
                           n_estimators = 500,
                           class_weight = 'balanced',
                           criterion = 'entropy',
                           max_features=0.3,  
                           oob_score = True,
                           random_state = 34,
                           n_jobs = -1)
```

####**5.모델 해석 및 데이터 시각화**

-PDP,SHAP,Permutation Importances 를 이용한 모델 해석 및 시각화


### **EDA & Feature Engineering 과정(일부)**
![image](https://user-images.githubusercontent.com/59490892/118280855-8d014000-b507-11eb-808f-79bc309358eb.png)
![image](https://user-images.githubusercontent.com/59490892/118280892-97233e80-b507-11eb-8ed5-7cc588c068cc.png)

###**모델 해석(일부)**
![image](https://user-images.githubusercontent.com/59490892/118280656-6216ec00-b507-11eb-99b7-ba902bee30f1.png)
![image](https://user-images.githubusercontent.com/59490892/118280675-66430980-b507-11eb-8276-c7666398ca3d.png)

