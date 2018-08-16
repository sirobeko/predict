import configparser
import pandas as pd
import oandapy
import datetime
from datetime import datetime, timedelta
import pytz
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt

def iso_to_jp(iso):
    date = None
    try:
        date = datetime.strptime(iso, '%Y-%m-%dT%H:%M:%S.%fZ')
        date = pytz.utc.localize(date).astimezone(pytz.timezone("Asia/Tokyo"))
    except ValueError:
        try:
            date = datetime.strptime(iso, '%Y-%m-%dT%H:%M:%S.%f%z')
            date = date.astimezone(pytz.timezone("Asia/Tokyo"))
        except ValueError:
            pass
    return date

def date_to_str(date):
    if date is None:
        return ''
    return date.strftime('%Y/%m/%d %H:%M:%S')

config = configparser.ConfigParser()
config.read('config/oanda_config.txt')
account_id = config['oanda']['account_id']
api_key = config['oanda']['api_key']

oanda = oandapy.API(environment = 'practice',
                    access_token = api_key)

#過去レート
res_hist = oanda.get_history(instrument = 'USD_JPY',
                             granularity = "M1",count = '5000')
res_hist_df = pd.DataFrame(res_hist['candles'])
mss = MinMaxScaler()
price_data = res_hist_df['openAsk']
price_data = price_data.reshape(-1,1)
input_dataframe = pd.DataFrame(mss.fit_transform(price_data))

def _load_data(data, n_prev=50):
    docX, docY = [], []
    for i in range(len(data)-n_prev):
        docX.append(data.iloc[i:i+n_prev].as_matrix())
        docY.append(data.iloc[i+n_prev].as_matrix())
    alsX = np.array(docX)
    alsY = np.array(docY)
    return alsX, alsY

def train_test_split(df, test_size=0.1, n_prev=50):
    ntrn = round(len(df) * (1 - test_size))
    ntrn = int(ntrn)
    X_train, y_train = _load_data(df.iloc[0:ntrn], n_prev)
    X_test, y_test = _load_data(df.iloc[ntrn:], n_prev)
    return (X_train, y_train), (X_test, y_test)

(X_train, y_train), (X_test, y_test) = train_test_split(input_dataframe)

in_out_neurons = 1
hidden_neurons = 300
length_of_sequences = 50

model = Sequential()
model.add(LSTM(hidden_neurons, batch_input_shape=(None, length_of_sequences, in_out_neurons), return_sequences=False))
model.add(Dense(in_out_neurons))
model.add(Activation("linear"))
model.compile(loss="mean_squared_error", optimizer="adam",)

early_stopping = EarlyStopping(monitor='val_loss', mode='auto', patience=0)
history = model.fit(X_train, y_train, batch_size=600, epochs=10, validation_split=0.1, callbacks=[early_stopping])

from matplotlib import pyplot as plt

pred_data = model.predict(X_train)
plt.plot(y_train, label='train')
plt.plot(pred_data, label='pred')
plt.legend(loc='upper left')
plt.show()
