import configparser
import pandas as pd
import oandapy
import datetime
from datetime import datetime, timedelta
import pytz

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

#現在レート
res = oanda.get_prices(instruments = 'USD_JPY')
val = res['prices'][0]
val['time'] = date_to_str((iso_to_jp(val['time'])))
print(*list(val.values()))
print()

#現在レート複数
res_mlt = oanda.get_prices(instruments = 'USD_JPY,EUR_JPY,GBP_JPY')
val2 = res_mlt['prices']
for i in range(len(val2)):
    val2[i]['time'] = date_to_str((iso_to_jp(val2[i]['time'])))
    print(*list(val2[i].values()))
    
#過去レート
res_hist = oanda.get_history(instrument = 'USD_JPY',
                             granularity = "M1",count = '5000')
res_hist_df = pd.DataFrame(res_hist['candles'])
res_hist_df['time'] = res_hist_df['time'].apply(lambda x:
                                                date_to_str(iso_to_jp(x)))
print(res_hist_df.head())
print(res_hist_df.shape)
