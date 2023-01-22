from statistics import mean, stdev
from typing import Any, Dict, Pattern, Set, Union, List
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import yfinance as yf


def prepare_prediction_data(open, close, high, low, volume, week_volume, month_volume, year_volume, week_close, month_close, year_close, close_day_before, close_week_before, close_month_before, close_year_before):
    '''
    POPRZEDNI DZIEŃ:
    - cena otwarcia
    - cena zamknięcia
    - najwyższa cena
    - najniższa cena
    - ilość transakcji
    - OSTATNI tydzień:
    - cena zamknięcia
    - ilość transakcji
    - OSTATNI MIESIĄC:
    - cena zamknięcia
    - ilość transakcji
    - OSTATNI ROK:
    - cena zamknięcia
    - ilość transakcji
    '''
    df_new = pd.DataFrame(index=[1])

    # 6 oryginalnych cech
    df_new['open'] = open
    df_new['open_1'] = open
    df_new['close_1'] = close
    df_new['high_1'] = high
    df_new['low_1'] = low
    df_new['volume_1'] = volume
    # 31 wygenerowanych cech
    # Średnie ceny
    df_new['avg_price_5'] = mean(week_close[1:])
    df_new['avg_price_30'] = mean(month_close[1:])
    df_new['avg_price_365'] = mean(year_close[1:])
    df_new['ratio_avg_price_5_30'] = df_new['avg_price_5'] / df_new['avg_price_30']
    df_new['ratio_avg_price_5_365'] = df_new['avg_price_5'] / df_new['avg_price_365']
    df_new['ratio_avg_price_30_365'] = df_new['avg_price_30'] / df_new['avg_price_365']
    # Średnie woluminy
    df_new['avg_volume_5'] = mean(week_volume)
    df_new['avg_volume_30'] = mean(month_volume)
    df_new['avg_volume_365'] = mean(year_volume)
    df_new['ratio_avg_volume_5_30'] = df_new['avg_volume_5'] / df_new['avg_volume_30']
    df_new['ratio_avg_volume_5_365'] = df_new['avg_volume_5'] / df_new['avg_volume_365']
    df_new['ratio_avg_volume_30_365'] = df_new['avg_volume_30'] / df_new['avg_volume_365']
    # Odchylenia standardowe cen
    df_new['std_price_5'] = stdev(week_close[1:])
    df_new['std_price_30'] = stdev(month_close[1:])
    df_new['std_price_365'] = stdev(year_close[1:])
    df_new['ratio_std_price_5_30'] = df_new['std_price_5'] / df_new['std_price_30']
    df_new['ratio_std_price_5_365'] = df_new['std_price_5'] / df_new['std_price_365']
    df_new['ratio_std_price_30_365'] = df_new['std_price_30'] / df_new['std_price_365']
    # Odchylenia standardowe woluminów
    df_new['std_volume_5'] = stdev(week_volume)
    df_new['std_volume_30'] = stdev(month_volume)
    df_new['std_volume_365'] = stdev(year_volume)
    df_new['ratio_std_volume_5_30'] = df_new['std_volume_5'] / df_new['std_volume_30']
    df_new['ratio_std_volume_5_365'] = df_new['std_volume_5'] / df_new['std_volume_365']
    df_new['ratio_std_volume_30_365'] = df_new['std_volume_30'] / df_new['std_volume_365']
    # Zwroty
    df_new['return_1'] = (close - close_day_before) / close_day_before
    df_new['return_5'] = (close - close_week_before) / close_week_before
    df_new['return_30'] = (close - close_month_before) / close_month_before
    df_new['return_365'] = (close - close_year_before) / close_year_before

    week_close_return = [((x - week_close[index-1]) / week_close[index-1]) if index-1 >= 0 else 0 for index, x in enumerate(week_close)]
    df_new['moving_avg_5'] = mean(week_close_return[1:])

    month_close_return = [((x - month_close[index-1]) / month_close[index-1]) if index-1 >= 0 else 0 for index, x in enumerate(month_close)]
    df_new['moving_avg_30'] = mean(month_close_return[1:])

    year_close_return = [((x - year_close[index-1]) / year_close[index-1]) if index-1 >= 0 else 0 for index, x in enumerate(year_close)]
    df_new['moving_avg_365'] = mean(year_close_return[1:])
    return df_new



def predict_next_day_price(data_to_scale):
    model = keras.models.load_model('models/final_model')
    gspc = yf.Ticker("^GSPC")
    data_raw = gspc.history(period="1y")
    data_raw = data_raw[["Open", "High", "Low", "Close", "Volume"]]

    open = data_raw["Open"].iloc[-1].item()
    close = data_raw["Close"].iloc[-1].item()
    high = data_raw["High"].iloc[-1].item()
    low = data_raw["Low"].iloc[-1].item()

    volume = data_raw["Volume"].iloc[-1].tolist()
    week_volume = data_raw["Volume"].iloc[-5:].tolist()
    month_volume = data_raw["Volume"].iloc[-21:].tolist()
    year_volume = data_raw["Volume"].iloc[-252:].tolist()

    week_close = data_raw["Close"].iloc[-6:].tolist()
    month_close = data_raw["Close"].iloc[-22:].tolist()
    year_close = data_raw["Close"].iloc[-253:].tolist()

    close_day_before = data_raw["Close"].iloc[-2].item()
    close_week_before = data_raw["Close"].iloc[-5].item()
    close_month_before = data_raw["Close"].iloc[-21].item()
    close_year_before = data_raw["Close"].iloc[-251].item()

    predictor_df = prepare_prediction_data(open, close, high, low, volume, week_volume, month_volume, 
    year_volume, week_close, month_close, year_close, close_day_before, close_week_before, 
    close_month_before, close_year_before)

    scaler = StandardScaler()
    scaler.fit_transform(data_to_scale)
    scaled_data = scaler.transform(predictor_df)

    return model.predict(scaled_data).item()
