
# 데이터

# 2/ 상업용 상대가격(기준=2015).csv
# 301열(인덱스포함) 5열(YEAR / MONTH / RP(상대가격) / GAS_PRICE(산업용도시가스), OIL_PRICE(원유정제처리제품)
# 1996.1~2020.12

# 3/ 월별공급량및비중.csv
# 301열(인덱스포함)  7열
# 1996.1~2020.12

# 4/ 제조업 부가가치(분기별).csv
# 101행(인덱스포함) 3열
# 1996~2020 (Q1~Q4)


# 1996.01~2020.12 까지의 데이터를 모두 하나의 파일로 만들어야함.
# 제조업 부가가치 데이터의 분기별 데이터를 월별로 증폭

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv1D, Conv2D, Flatten, MaxPooling2D, Dropout, LSTM
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


#1. 데이터
path = './data/'
train_set = pd.read_csv(path + 'train.csv', # train.csv 의 데이터가 train set에 들어가게 됨
                        index_col=0) # 0번째 컬럼은 인덱스로 지정하는 명령
#print(train_set)
#print(train_set.shape) # (1460, 80) 원래 열이 81개지만, id를 인덱스로 제외하여 80개

test_set = pd.read_csv(path + 'test.csv',
                       index_col=0)
x = 
y =

print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.8, shuffle=True, random_state=123)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) # 수치로 변환해주는 걸 x_train에 집어넣자.
x_test = scaler.transform(x_test) 

# 2차원의 경우 3차원으로 reshape을 해주고
# (1,2,3) -> input_shape(2,3)을 넣어줌


#2. 모델구성
model = Sequential()
# model.add(LSTM(10, input_shape=(3,1), return_sequences=False))
model.add(Conv1D(10, 2, input_shape=(3,1))) # filter / kenner_size / input_shape 순서로 써줌
model.add(Flatten())
model.add(Dense(3, activation='relu'))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2 스코어 : ', r2)