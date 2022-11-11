
import numpy as np 
import pandas as pd 
#from xgboost import XGBRegressor 
from catboost import CatBoostRegressor # 가상환경 azureml_py38 
import catboost as cgb 
import matplotlib.pyplot as plt 
import seaborn as sns 
import matplotlib
import platform 
from matplotlib import font_manager, rc 
from sklearn.preprocessing import StandardScaler, RobustScaler 
matplotlib.rcParams['font.family']='gulim'
# if platform.system() 'Windows': 
#      font_name font_manager. Font Properties(fname="C:\Windows\Fonts\Malgun, ttf").get_name() 
#      rc('font, family-font_name)
matplotlib.rcParams['axes.unicode_minus']=False 
from xgboost import XGBRegressor

# from keras.models import Sequential 
# from keras.layers import ConviD, MaxPooling10, Dense, GlobalAverage Pooling1D 
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Dense, ConviD, MaxPooling10,GlobalAveragePooling1D, Flatten 
# from tensorflow.python.keras. layers import LSTM, GRU 
# from sklearn.model_selection import train_test_split 
# from sklearn.model_selection import RandomizedSearchCV 
# from sklearn.metrics import r2_score,mean_absolute_error 
# from xgboost impo
# Data Load 
path2 = 'Z:\data\sample_dataset/'
path = 'Z:\\test/'
data_a = pd.read_csv(path2+'월별공급량비중.csv')
data_b = pd.read_csv(path2+'상업용 상대가격(기준=2015).csv')
data_c = pd.read_csv(path2+'제조업 부가가치(분기별).csv') 
data_d = pd.read_csv(path+'추가데이터(1).csv', thousands = ',')

# 데이터별 인덱스가 상이하여 맞춰주는 작업 및 이름 변경
data_c = data_c.rename(columns={'QVA(제조업부가가치/단위:십억원)':'MFG'})
# 분기로 합쳐져있는 값 분산하기
# print(data_c['MFG'])

data_c['MFG'] = np.round(data_c[['MFG']].transform(lambda x : x / 3),0)
data_c['MFG'] = np.round(data_c[['MFG']].transform(lambda x : x / 3),0)
data_c = pd.concat([data_c, data_c, data_c], ignore_index=True)
data_c = pd.concat([data_c, data_c, data_c], ignore_index=True)
print(len(data_c))
# data_c = data_c.sort_values(by=['QUARTER'], axis=0, ascending=False)
data_c = data_c.sort_values(by=['YEAR', 'QUARTER'], axis=0, ascending=True)
idx = pd.date_range('01-01-1996', '12-31-2020', freq='M')
print(len(idx)) 
print(len(data_c)) 
'''
data_c['date'] = pd.to_datetime (idx)
data_c['MONTH'] = data_c['date'].dt.month
data_c = data_c.drop(['date'], axis=1)
data_c = data_c.reset_index(drop=True)
data_c = data_c[['YEAR', 'MONTH', 'MFG, 'GDP', 'QUARTER']]

# 데이터 병합
data_all = pd.merge(data_a, data_b) 
data_all = data_all.merge(data_c)
data_all = data_all.merge(data_d) 
# print(data_all.columns)

# 분류형 피처 원핫처리
data_all = pd.get_dummies(data_all, columns=['QUARTER'])
data_all = data_all.rename(
    columns={'도시가스(톤)_민수용':'CIVIL', 'RP(상대가격)':'RP',
    '도시가스(톤)_산업용':'IND', '민수용비중' : 'CIVILper',
    '산업용비중':'INDper', 'GAS PRICE(산업용도시가스)': 'GAS_PRICE',
    'OIL_PRICE(원유정제처리제품)': 'OIL PRICE', '평균기온':'Meantemp',
    '난방도일':'nanbangdoil', '냉방도일':'naengbangdoil',
    '최저기온':'lowtemp', '최고기온': 'hightemp',
    '천연가스생산량(백만): 'amount_of_gas', '산업소비량(백만)':'INDcon',
    '가정소비량(백만)':'CIVILcon', '도시가스(톤)_종합(민수용+산업용)':'Total'})
    
# print(data_all.columns) 

#전월비 증감량
data_all['MOM_CIVIL'] = data_all['CIVIL'].pct_change()
data_all['MOM IND'] = data_all['IND'].pct_change() 
# data_all['MOM_GDP'] = data_all['GDP'].pct_change() 
# print(data_all.corr()) 
# sns.set(font_scale=0.8) 
# sns.heatmap(data=data_all.corr(), square=True, annot=True, cbar=True) 
# plt.show()

# data_all = data_all.drop(['도시가스(톤)_총합(민수용+산업용), 'RP(상대가격)'], axis=1) #39531 
# data_all = data_all.drop(['RP', 'Total', 'hightemp', 'lowtemp'], axis=1) #39531 
data_all = data_all.drop(['RP', 'Total', 'amount_of_gas', 'hightemp"], axis=1) #39531 
# data_all = data_all.drop(['amount_of_gas', 'lowtemp", "hightemp', 'nanbangdoil'], axis=1) #39531 
#Index(['YEAR', 'MONTH', 'CIVIL', 'IND', 'CIVILper', 'INDper', 'GAS_PRICE', 
#       'OIL_PRICE', 'MFG', 'GDP', 'Meantemp', 'naengbangdoil', 'INDcon', 
#       'CIVILCON', 'QUARTER_Q1', 'QUARTER_Q2', 'QUARTER_Q3', 'QUARTER_Q4,
#       'MOM_CIVIL', 'MOM_IND'] 
# print(data_all.columns)
'''

'''
size = 48 
def split_x(dataset, size): 
    aaa = [] 
    for i in range(len(dataset) - size +1): 
        subset = dataset[i: (i + size)] 
        aaa.append(subset) 
    return np.array(aaa) 
    
y = data_all[['CIVIL', 'IND']]
# y2 data_all 
x = data_all 
# print(x.shape, y.shape) # (380, 9) (300, 2) 

bbb = split_x(y, size)
ccc = split_x(x, size) 

x1 = ccc[:, :24]
y1 = bbb[:, 24:] 

print(x1.shape, y1.shape) # (121, 12, 11) (121, 168, 2) 
# print(x_test.shape,y_test.shape) # (168, 3, 9) (168, 2) 

x1 = x1.reshape(x1.shape[0]*x1.shape[1], x1.shape[2])
y1 = y1.reshape(y1.shape[0]*y1.shape[1], y1.shape[2]) 
# y1 = np.logip(y1)
# print(x1.shape, y1.shape) # (3324, 9) (3324, 2) 
# (1452, 11) (20328, 2)

 
from sklearn.model_selection import train_test_split, KFold 
x_train, x_test, y_train, y_test =  train_test_split(x1, y1, 
    shuffle=False, train_size=0.93)

scaler RobustScaler())
# x_train = scaler.fit_transform(x_train) 
# x_test = scaler.transform(x_test) 
# y_train = np.log1p(y_train)
# shuffle True로?
n_splits = 5 
# 5 
# 6 31135.829668209877 
# 7 31135.829668209877 
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)
print(x_train.shape, y_train.shape) # (3324, 9) (3324, 2) 
print(x_test.shape, y_test.shape) # (831, 9) (831, 2)
'''
import optuna # 가상환경 azureml_py38
from optuna import Trial, visualization 
from optuna.samplers import TPESampler 
# from sklearn.metrics import square_mean_absolute_error


###### parameter data type
# optuna.trial.Trial.suggest_categorical(): 리스트 범위 내에서 값을 선택한다.
# optuna.trial.Trial.suggest_int():
# optuna.trial.Trial.suggest_float():
# optuna.trial.Trial.suggest_uniform():
# optuna.trial.Trial.suggest_discrete_uniform(): 
# optuna.trial.Trial.suggest_loguniform(): 
############################
# Best trial: score 72836.92161800986, 
# params {'n_estimators': 377, 'max_depth': 7, 
# 'gamma': 0.6218549737305442, 'learning_rate': 0.10967344525838242, 
# 'subsample': 0.06251714312171233, 'colsample_bytree': 0.8242609278359663, 
# 'colsample_bylevel': 0.09859872463675436, 'reg_lambda': 7.444274223309037,
# 'reg_alpha': 1.2522340205973888, 'random_state': 96)

import time 

# Best trial: score 58792.737847697374, 
# params {'n_estimators': 377, 'max_depth': 7, 
# 'gamma': 0.4902966798659445, 'learning rate': 0.02999201956430698, 
# 'subsample': 0.13495898779130204, 'colsample_bytree': 0.7662915180839261,
# 'colsample_bylevel': 0.03499217955446093, 'reg lambda': 7.558994217238986, 
# 'reg_alpha': 4.6289329141354365, 'random_state': 100}
# 걸린 시간 : 843.295622587204

'''

start_time time.time() 
# optuna 적용해서 최적 파라미터 찾기
def objectiveXGB(trial: Trial, x_train, y_train, x_test):
    param = {
        'n_estimators' : trial.suggest_int('n_estimators', 100, 1000),
        'max_depth' : trial.suggest_int('max_depth', 5, 10),
        'gamma' : trial.suggest_float('gamma', 0.3, 1),
        'learning_rate': trial.suggest_float('learning rate', 0.001, 0.06),         
        'subsample' : trial.suggest_float('subsample', 0, 1),
        'colsample_bytree' : trial.suggest_float('colsample_bytree', 0, 1), 
        'colsample_bylevel' : trial.suggest_float('colsample_bylevel', 0, 1),
        'reg lambda' : trial.suggest_float('reg lambda', 0, 10),
        'reg_alpha' : trial.suggest_float('reg_alpha', 0, 10),
        'random_state' : trial.suggest_int('random_state', 72, 72),
    }
    
    # 학습 모델 생성
    model = XGBRegressor(**param)
    XGB_model = model.fit(x_train, y_train, verbose=True) # 학습 진행
    
    # 모델 성능 확인
    score = mean_absolute_error(y_test, XGB_model.predict(x_test))
    # score = r2_score(CAT_model.predict(x_test), y_test)
    
    return score 
    
# MAE가 최소가 되는 방향으로 학습을 진행
# TPESampler : Sampler using TPE (Tree-structured Parzen Estimator) algorithm. 
study = optuna.create_study(direction='minimize', sampler=TPESampler()) 

# n_trials 지정해주지 않으면, 무한반복
study.optimize(lambda trial : objectiveXGB(trial, x, y, x_test), n_trials = 3000)

print('Best trial: score {}, \nparms {}'.format(study.best_trial.value, study.best_trial.params))
print('걸린 시간 :', time.time()-start_time)
'''

# 하이퍼파라미터별 중요도를 확인할 수 있는 그래프
# print(optuna.visualization.plot_param_importances (study)) 

# 하이퍼파라미터 최적화 과정을 확인
# optuna.visualization.plot_optimization_history(study)

# Best trial: score 72836.92161800986, 
# params {'n_estimators': 194, 'max_depth: 7,
# 'gamma': 0.6218549737305442, 'learning_rate': 0.18967344525838242,
# 'subsample': 0.06251714312171233, 'colsample_bytree': 0.8242609278359663, 
# 'colsample_bylevel': 0.09859872463675436, 'reg lambda': 7.444274223309037, 
# 'reg_alpha': 1.2522340205973888, 'random_state': 96}

# Best trial: score 70983.56684646382, 
# params {'n_estimators': 160, 'max_depth': 7, 
# 'gamma': 0.6120521563886373, 'learning_rate": 0.11490493295493891, 
# 'subsample': 0.06916655646029328, 'colsample_bytree': 0.1488309493804681, 
# 'colsample_bylevel': 0.6981257882206031, 'reg lambda': 9.803477165210557, 
# 'reg_alpha': 1.3936008844131718, 'random_state': 197} 

# Best trial: score 58792.737047697374, 
# params {'n_estimators': 377, 'max_depth': 7, 
# 'gamma': 0.4902966798659445, 'learning rate': 0.02999201956430698, 
# 'subsample': 0.13495890779130204, 'colsample_bytree': 0.7662915188839261, 
# 'colsample_bylevel': 0.03499217955446093, 'reg lambda': 7.558994217238986, 
# 'reg_alpha': 4.6289329141354365, 'random_state': 100} 
# 걸린 시간 : 843.295622587284 

'''
# MAE: 39531.3335988898 파라미터
parameters = {'n_estimators': [843],
                'max_depth': [8],
                'gamma': [0.4792690178653076],
                'learning rate': [0.014157532049638313],
                'subsample': [0.11071806398348982], 
                'colsample_bytree': [0.20103050072750606], 
                'colsample_bylevel': [0.1569952254098891], 
                'reg lambda': [5.284833949525178], 
                'reg_alpha': [2.674562099242543], 'random_state': [123]}

#2. 모델구성

xgb = XGBRegressor() 
model = RandomizedSearchCV(xgb, parameters, cv=kfold, n_jobs=-1)
start_time2 = time.time() 

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
results = model.score(x_test,y_test) 
print('score', results) 
y_pred model.predict(x_test)

y123 = y_pred[-24:]
# print('2821-01~12', y123.shape) # (831, 2) 
idx = 6 
L = []
for i in range(idx):

    y12 = model.predict(x_test[-24:])
    L.append(y12)
L = np.array(L).reshape(-1,2)
y123 = np.concatenate([y123, L])

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)


submission = pd.read_csv(path + 'submission_sample.csv')
# CIVIL, IND 
submission['CIVIL'] = y123[:0]
submission['IND'] = y123[,1]

# print(submission)
import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
submission.to_csv('try-'+date+'.csv', index=True)
print('==========')
print('DATE : ', date)
print('R2 : ', r2)
print('MAE : ', mae)
print('걸린 시간 : ", time.time()-start_time2)
print('Done')
print('==========')
'''