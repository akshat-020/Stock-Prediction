import requests
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from pandas_datareader import data as pdr
import yfinance as yf
from tensorflow import keras
import tensorflow as tf
import plotly.express as px
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

user_input=st.text_input('Enter Stock Ticker', 'IBM')


yf.pdr_override() # <== that's all it takes :-)
# download dataframe
data_IBM = pdr.get_data_yahoo(user_input, start="2008-01-01", end="2022-04-30")
len(data_IBM)


st.write(data_IBM.describe())

#Visualize

st.subheader('Closing Price v/s Time Chart')
fig=plt.figure(figsize=(12,6))
plt.plot(data_IBM.Close)
st.pyplot(fig)

st.subheader('Closing Price v/s Time Chart with 100MA')
ma100=data_IBM.Close.rolling(100).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(data_IBM.Close)
st.pyplot(fig)


st.subheader('Closing Price v/s Time Chart with 100MA and 200MA')
ma200=data_IBM.Close.rolling(200).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(data_IBM.Close)
st.pyplot(fig)








url = 'https://www.alphavantage.co/query?function=RSI&symbol=IBM&interval=daily&time_period=14&series_type=open&apikey=V23MOD5YM3ZNW9OR'
r = requests.get(url)
rsi_data = r.json()

#print(rsi_data)
new_rsi= pd.DataFrame.from_dict(rsi_data['Technical Analysis: RSI'])
new1_rsi=new_rsi.transpose()
new1_rsi=new1_rsi.iloc[::-1].loc["2008-01-01":"2022-04-30", :]

url = 'https://www.alphavantage.co/query?function=SMA&symbol=IBM&interval=daily&time_period=14&series_type=open&apikey=V23MOD5YM3ZNW9OR'
r = requests.get(url)
sma_data = r.json()

#print(sma_data)
new_sma= pd.DataFrame.from_dict(sma_data['Technical Analysis: SMA'])
new1_sma=new_sma.transpose()
new1_sma=new1_sma.iloc[::-1].loc["2008-01-01":"2022-04-30", :]

# replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
url = 'https://www.alphavantage.co/query?function=EMA&symbol=IBM&interval=daily&time_period=10&series_type=open&apikey=V23MOD5YM3ZNW9OR'
r = requests.get(url)
ema_data = r.json()

#print(ema_data)
#print(ema_data['Technical Analysis: EMA'])

new_ema= pd.DataFrame.from_dict(ema_data['Technical Analysis: EMA'])
new1_ema=new_ema.transpose()
new1_ema=new1_ema.iloc[::-1].loc["2008-01-01":"2022-04-30", :]

url = 'https://www.alphavantage.co/query?function=ADX&symbol=IBM&interval=daily&time_period=14&apikey=V23MOD5YM3ZNW9OR'
r = requests.get(url)
adx_data = r.json()

#print(adx_data)
#print(adx_data['Technical Analysis: ADX'])

new_adx= pd.DataFrame.from_dict(adx_data['Technical Analysis: ADX'])
new1_adx=new_adx.transpose()
new1_adx=new1_adx.iloc[::-1].loc["2008-01-01":"2022-04-30", :]

url = 'https://www.alphavantage.co/query?function=OBV&symbol=IBM&interval=daily&apikey=V23MOD5YM3ZNW9OR'
r = requests.get(url)
obv_data = r.json()

#print(obv_data)
new_obv= pd.DataFrame.from_dict(obv_data['Technical Analysis: OBV'])
new1_obv=new_obv.transpose()
new1_obv=new1_obv.iloc[::-1].loc["2008-01-01":"2022-04-30", :]
new1_obv.index.rename('Date', inplace=True)


data_IBM=data_IBM.loc["2008-01-01":"2022-04-30",:]
print(data_IBM.head(3))
data_IBM = data_IBM.assign(EMA = list(new1_ema['EMA']))
data_IBM = data_IBM.assign(OBV = list(new1_obv['OBV']))
data_IBM = data_IBM.assign(SMA = list(new1_sma['SMA']))



data_IBM['Close'].plot()


def custom_ts_multi_data_prep(data_IBM, target,window, horizon,start,end):
     X = []
     y = []
     start = start + int(window)
     if end is None:
         end = len(data_IBM) - horizon
     for i in range(start, end):
         indices = range(i-window, i)
         X.append(data_IBM[indices])
         indicey = range(i+1, i+1+horizon)
         y.append(target[indicey])
     return np.array(X), np.array(y)
    

X_scaler = MinMaxScaler()
Y_scaler = MinMaxScaler()
X_data=X_scaler.fit_transform(data_IBM[['Low','High','Open','EMA','SMA','OBV']])
Y_data=Y_scaler.fit_transform(data_IBM[['Close']])

hist_window = 60
horizon = 1
TRAIN_SPLIT=int(0.8*len(data_IBM))
x_train, y_train = custom_ts_multi_data_prep(X_data, Y_data,hist_window, horizon,0, TRAIN_SPLIT )
x_vali, y_vali = custom_ts_multi_data_prep(X_data, Y_data,hist_window, horizon,TRAIN_SPLIT,None)

x_train=np.asarray(x_train).astype(np.float32)
y_train=np.asarray(y_train).astype(np.float32)
x_vali=np.asarray(x_vali).astype(np.float32)
y_vali=np.asarray(y_vali).astype(np.float32)


train_x=tf.convert_to_tensor(x_train)
train_y=tf.convert_to_tensor(y_train)
vali_x=tf.convert_to_tensor(x_vali)
vali_y=tf.convert_to_tensor(y_vali)



model = keras.Sequential()
model.add(keras.layers.Bidirectional(keras.layers.LSTM(units=50,activation='tanh',return_sequences=True,input_shape=(x_train.shape[-2:]))))


model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.LSTM(units=60,activation='tanh',return_sequences=True))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.LSTM(units=80,activation='tanh',return_sequences=True))
model.add(keras.layers.Dropout(0.4))
model.add(keras.layers.LSTM(units=100,activation='tanh'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(1))
# Define optimizer and metric for loss function
model.compile(optimizer='adam',loss='mean_squared_error')


history =model.fit(train_x,train_y,epochs=10,batch_size=64,verbose=1)
# Run model
x = np.array(vali_y).reshape(vali_y.shape[0],horizon)
x_inverse = Y_scaler.inverse_transform(x)

vali_predict=model.predict(vali_x,batch_size=64)
#val_rescaled = data_val.reshape(1, data_val.shape[0], data_val.shape[1])
pred_Inverse = Y_scaler.inverse_transform(vali_predict)



x_inverse.reshape(661,)
ind = np.linspace(1,661, 661)
df = pd.DataFrame(list(zip(ind, x_inverse.reshape(661,), pred_Inverse.reshape(661,))),
               columns =['ind','true', 'predicted'])


#fig = px.line(df, x="ind", y=["true", 'predicted'])

#fig.show()
#st.plotly_chart(fig, use_container_width=True)




hist_window = 60
horizon = 15
TRAIN_SPLIT=int(0.8*len(data_IBM))
x_train_for, y_train_for = custom_ts_multi_data_prep(X_data, Y_data,hist_window, horizon,0, TRAIN_SPLIT )
x_vali_for, y_vali_for = custom_ts_multi_data_prep(X_data, Y_data,hist_window, horizon,TRAIN_SPLIT,None) 

x_train_for=np.asarray(x_train_for).astype(np.float32)
y_train_for=np.asarray(y_train_for).astype(np.float32)
x_vali_for=np.asarray(x_vali_for).astype(np.float32)
y_vali_for=np.asarray(y_vali_for).astype(np.float32)


train_x_for=tf.convert_to_tensor(x_train_for)
train_y_for=tf.convert_to_tensor(y_train_for)
vali_x_for=tf.convert_to_tensor(x_vali_for)
vali_y_for=tf.convert_to_tensor(y_vali_for)


# Build model. Include 20% drop out to minimize overfitting
model_for = keras.Sequential()
model_for.add(keras.layers.Bidirectional(keras.layers.LSTM(units=50,activation='tanh',return_sequences=True,input_shape=(x_train.shape[-2:]))))


model_for.add(keras.layers.Dropout(0.2))
model_for.add(keras.layers.LSTM(units=60,activation='tanh',return_sequences=True))
model_for.add(keras.layers.Dropout(0.3))
model_for.add(keras.layers.LSTM(units=80,activation='tanh',return_sequences=True))
model_for.add(keras.layers.Dropout(0.4))
model_for.add(keras.layers.LSTM(units=100,activation='tanh'))
model_for.add(keras.layers.Dropout(0.5))
model_for.add(keras.layers.Dense(15))
# Define optimizer and metric for loss function
model_for.compile(optimizer='adam',loss='mean_squared_error')


history_for =model_for.fit(train_x,train_y,epochs=10,batch_size=64,verbose=1)
# Run model

x_input=X_data[-hist_window:,]
x_input=x_input.reshape((1,x_input.shape[0],x_input.shape[1]))
yhat=model_for.predict(x_input,verbose=0)
yhat= Y_scaler.inverse_transform(yhat)

np.linspace(661, 675, 15)

df1 = pd.DataFrame(list(zip(np.linspace(661, 675, 15), yhat[0])),
               columns =['ind','forecast'])

df2 = pd.merge(df, df1, on = 'ind', how = 'outer')



fig = px.line(df2, x="ind", y=["true", 'predicted', 'forecast'])
st.plotly_chart(fig, use_container_width=True)