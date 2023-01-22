from dataclasses import dataclass
import os
import re
import base64
from pathlib import Path
from typing import Any, Dict, Pattern, Set, Union, List
import streamlit as st
from streamlit_option_menu import option_menu
import plotly.express as px
import pandas as pd
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from predictor import predict_next_day_price


@dataclass
class DatasetSplit:
    data_train: Any
    X_train: Any
    y_train: Any
    data_test: Any
    X_test: Any
    y_test: Any
    X_scaled_train: Any
    X_scaled_test: Any


def markdown_images(markdown):
    # example image markdown:
    # ![Test image](images/test.png "Alternate text")
    images = re.findall(r'(!\[(?P<image_title>[^\]]+)\]\((?P<image_path>[^\)"\s]+)\s*([^\)]*)\))', markdown)
    return images


def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded


def img_to_html(img_path, img_alt):
    img_format = img_path.split(".")[-1]
    img_html = f'<img src="data:image/{img_format.lower()};base64,{img_to_bytes(img_path)}" alt="{img_alt}" style="max-width: 100%;">'

    return img_html


def markdown_insert_images(markdown):
    images = markdown_images(markdown)

    for image in images:
        image_markdown = image[0]
        image_alt = image[1]
        image_path = image[2]
        if os.path.exists(image_path):
            markdown = markdown.replace(image_markdown, img_to_html(image_path, image_alt))
    return markdown


def rewrite_markdown_script(file: str):
    with open(file, "r") as description_file:
        markdown = description_file.read()

    return markdown_insert_images(markdown)


def plot_data(data_raw):
    fig = px.line(data_raw, x=data_raw.index, y=["Open", "Close"], line_shape="spline", labels={"variable":"Legenda", "value":"Cena"}, 
                render_mode="svg", title="Wykres wartości otwarcia i zamknięcia indeksu S&P500 w czasie")
    newnames = {"Open":"Cena otwarcia", "Close":"Cena zamknięcia"}
    fig.for_each_trace(lambda t: t.update(name = newnames[t.name],
                                        legendgroup = newnames[t.name],
                                        hovertemplate = t.hovertemplate.replace(t.name, newnames[t.name])
                                        )
                    )

    fig1 = px.area(data_raw, x=data_raw.index, y="Volume", line_shape="spline", 
                title="Wykres ilości transakcji indeksu S&P500 w czasie")
    
    tab1, tab2 = st.tabs(["Cena otwarcia-zamknięcia", "Ilość transakcji"])
    with tab1:
        st.plotly_chart(fig, theme=None, use_container_width=True)
    with tab2:
        st.plotly_chart(fig1, theme=None, use_container_width=True)


def get_dataset():
    gspc = yf.Ticker("^GSPC")
    data_raw = gspc.history(period="max")
    data_raw = data_raw[["Open", "High", "Low", "Close", "Volume"]]
    return data_raw["1982-04-20 00:00:00-05:00":]


def data_preparation(data_raw):
    data = generate_features(data_raw)

    start_train = '20-04-1982'
    end_train = '31-12-2021'
    start_test = '03-01-2022'
    end_test = '17-01-2023'

    data_train = data.loc[start_train:end_train]
    X_train = data_train.drop('close', axis=1).values
    y_train = data_train['close'].values
    data_test = data.loc[start_test:end_test]
    X_test = data_test.drop('close', axis=1).values
    y_test = data_test['close'].values

    scaler = StandardScaler()
    X_scaled_train = scaler.fit_transform(X_train)
    X_scaled_test = scaler.transform(X_test)

    prepared_data = DatasetSplit(data_train=data_train, X_train=X_train, y_train=y_train, 
    data_test=data_test, X_test=X_test, y_test=y_test, X_scaled_train=X_scaled_train, 
    X_scaled_test=X_scaled_test)
    
    return prepared_data


def train_default_NN(prepared_data):
    model = keras.models.load_model('models/default_model')
    predictions = model.predict(prepared_data.X_scaled_test)[:, 0]
    st.write(f'Błąd średniokwadratowy: {mean_squared_error(prepared_data.y_test, predictions):.3f}')
    st.write(f'Średni błąd bezwzględny: {mean_absolute_error(prepared_data.y_test, predictions):.3f}')
    st.write(f'R^2: {r2_score(prepared_data.y_test, predictions):.3f}')
    return predictions


def final_neural_network(prepared_data):
    model = keras.models.load_model('models/final_model')
    predictions = model.predict(prepared_data.X_scaled_test)[:, 0]
    st.write(f'Błąd średniokwadratowy: {mean_squared_error(prepared_data.y_test, predictions):.3f}')
    st.write(f'Średni błąd bezwzględny: {mean_absolute_error(prepared_data.y_test, predictions):.3f}')
    st.write(f'R^2: {r2_score(prepared_data.y_test, predictions):.3f}')
    return predictions


def predictions_plot(predictions, prepared_data):
    # Utworzenie wykresu wartości prognozowanych i rzeczywistych
    fig2 = px.line(x=prepared_data.data_test.index, y=[prepared_data.y_test, predictions], line_shape="spline", labels={"variable":"Legenda", "value":"Cena zamknięcia", "x":"Data"}, 
                render_mode="svg", title="Wykres wartości rzeczywistych i prognozowanych indeksu S&P500 w czasie")
    newnames = {"wide_variable_0":"Wartości rzeczywiste", "wide_variable_1":"Prognozy sieci neuronowej"}
    fig2.for_each_trace(lambda t: t.update(name = newnames[t.name],
                                        legendgroup = newnames[t.name],
                                        hovertemplate = t.hovertemplate.replace(t.name, newnames[t.name])
                                        )
                    )
    st.plotly_chart(fig2, theme=None, use_container_width=True)


def generate_features(df):
    """
    Funkcja generująca cechy na podstawie historycznych wartości indeksu i jego zmienności
    @param df: obiekt DataFrame zawierający kolumny "Open", "Close", "High", "Low", "Volume"
    @return: obiekt DataFrame zawierający zbiór danych z nowymi cechami    
    """
    df_new = pd.DataFrame()
    # 6 oryginalnych cech
    df_new['open'] = df['Open']
    df_new['open_1'] = df['Open'].shift(1)
    df_new['close_1'] = df['Close'].shift(1)
    df_new['high_1'] = df['High'].shift(1)
    df_new['low_1'] = df['Low'].shift(1)
    df_new['volume_1'] = df['Volume'].shift(1)
    # 31 wygenerowanych cech
    # Średnie ceny
    df_new['avg_price_5'] = df['Close'].rolling(5).mean().shift(1)
    df_new['avg_price_30'] = df['Close'].rolling(21).mean().shift(1)
    df_new['avg_price_365'] = df['Close'].rolling(252).mean().shift(1)
    df_new['ratio_avg_price_5_30'] = df_new['avg_price_5'] / df_new['avg_price_30']
    df_new['ratio_avg_price_5_365'] = df_new['avg_price_5'] / df_new['avg_price_365']
    df_new['ratio_avg_price_30_365'] = df_new['avg_price_30'] / df_new['avg_price_365']
    # Średnie woluminy
    df_new['avg_volume_5'] = df['Volume'].rolling(5).mean().shift(1)
    df_new['avg_volume_30'] = df['Volume'].rolling(21).mean().shift(1)
    df_new['avg_volume_365'] = df['Volume'].rolling(252).mean().shift(1)
    df_new['ratio_avg_volume_5_30'] = df_new['avg_volume_5'] / df_new['avg_volume_30']
    df_new['ratio_avg_volume_5_365'] = df_new['avg_volume_5'] / df_new['avg_volume_365']
    df_new['ratio_avg_volume_30_365'] = df_new['avg_volume_30'] / df_new['avg_volume_365']
    # Odchylenia standardowe cen
    df_new['std_price_5'] = df['Close'].rolling(5).std().shift(1)
    df_new['std_price_30'] = df['Close'].rolling(21).std().shift(1)
    df_new['std_price_365'] = df['Close'].rolling(252).std().shift(1)
    df_new['ratio_std_price_5_30'] = df_new['std_price_5'] / df_new['std_price_30']
    df_new['ratio_std_price_5_365'] = df_new['std_price_5'] / df_new['std_price_365']
    df_new['ratio_std_price_30_365'] = df_new['std_price_30'] / df_new['std_price_365']
    # Odchylenia standardowe woluminów
    df_new['std_volume_5'] = df['Volume'].rolling(5).std().shift(1)
    df_new['std_volume_30'] = df['Volume'].rolling(21).std().shift(1)
    df_new['std_volume_365'] = df['Volume'].rolling(252).std().shift(1)
    df_new['ratio_std_volume_5_30'] = df_new['std_volume_5'] / df_new['std_volume_30']
    df_new['ratio_std_volume_5_365'] = df_new['std_volume_5'] / df_new['std_volume_365']
    df_new['ratio_std_volume_30_365'] = df_new['std_volume_30'] / df_new['std_volume_365']
    # Zwroty
    df_new['return_1'] = ((df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1)).shift(1)
    df_new['return_5'] = ((df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5)).shift(1)
    df_new['return_30'] = ((df['Close'] - df['Close'].shift(21)) / df['Close'].shift(21)).shift(1)
    df_new['return_365'] = ((df['Close'] - df['Close'].shift(252)) / df['Close'].shift(252)).shift(1)
    df_new['moving_avg_5'] = df_new['return_1'].rolling(5).mean().shift(1)
    df_new['moving_avg_30'] = df_new['return_1'].rolling(21).mean().shift(1)
    df_new['moving_avg_365'] = df_new['return_1'].rolling(252).mean().shift(1)
    # Wartości docelowe
    df_new['close'] = df['Close']
    df_new = df_new.dropna(axis=0)
    return df_new


generate_features_function = '''
def generate_features(df):
    """
    Funkcja generująca cechy na podstawie historycznych wartości indeksu i jego zmienności
    @param df: obiekt DataFrame zawierający kolumny "Open", "Close", "High", "Low", "Volume"
    @return: obiekt DataFrame zawierający zbiór danych z nowymi cechami    
    """
    df_new = pd.DataFrame()
    # 6 oryginalnych cech
    df_new['open'] = df['Open']
    df_new['open_1'] = df['Open'].shift(1)
    df_new['close_1'] = df['Close'].shift(1)
    df_new['high_1'] = df['High'].shift(1)
    df_new['low_1'] = df['Low'].shift(1)
    df_new['volume_1'] = df['Volume'].shift(1)
    # 31 wygenerowanych cech
    # Średnie ceny
    df_new['avg_price_5'] = df['Close'].rolling(5).mean().shift(1)
    df_new['avg_price_30'] = df['Close'].rolling(21).mean().shift(1)
    df_new['avg_price_365'] = df['Close'].rolling(252).mean().shift(1)
    df_new['ratio_avg_price_5_30'] = df_new['avg_price_5'] / df_new['avg_price_30']
    df_new['ratio_avg_price_5_365'] = df_new['avg_price_5'] / df_new['avg_price_365']
    df_new['ratio_avg_price_30_365'] = df_new['avg_price_30'] / df_new['avg_price_365']
    # Średnie woluminy
    df_new['avg_volume_5'] = df['Volume'].rolling(5).mean().shift(1)
    df_new['avg_volume_30'] = df['Volume'].rolling(21).mean().shift(1)
    df_new['avg_volume_365'] = df['Volume'].rolling(252).mean().shift(1)
    df_new['ratio_avg_volume_5_30'] = df_new['avg_volume_5'] / df_new['avg_volume_30']
    df_new['ratio_avg_volume_5_365'] = df_new['avg_volume_5'] / df_new['avg_volume_365']
    df_new['ratio_avg_volume_30_365'] = df_new['avg_volume_30'] / df_new['avg_volume_365']
    # Odchylenia standardowe cen
    df_new['std_price_5'] = df['Close'].rolling(5).std().shift(1)
    df_new['std_price_30'] = df['Close'].rolling(21).std().shift(1)
    df_new['std_price_365'] = df['Close'].rolling(252).std().shift(1)
    df_new['ratio_std_price_5_30'] = df_new['std_price_5'] / df_new['std_price_30']
    df_new['ratio_std_price_5_365'] = df_new['std_price_5'] / df_new['std_price_365']
    df_new['ratio_std_price_30_365'] = df_new['std_price_30'] / df_new['std_price_365']
    # Odchylenia standardowe woluminów
    df_new['std_volume_5'] = df['Volume'].rolling(5).std().shift(1)
    df_new['std_volume_30'] = df['Volume'].rolling(21).std().shift(1)
    df_new['std_volume_365'] = df['Volume'].rolling(252).std().shift(1)
    df_new['ratio_std_volume_5_30'] = df_new['std_volume_5'] / df_new['std_volume_30']
    df_new['ratio_std_volume_5_365'] = df_new['std_volume_5'] / df_new['std_volume_365']
    df_new['ratio_std_volume_30_365'] = df_new['std_volume_30'] / df_new['std_volume_365']
    # Zwroty
    df_new['return_1'] = ((df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1)).shift(1)
    df_new['return_5'] = ((df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5)).shift(1)
    df_new['return_30'] = ((df['Close'] - df['Close'].shift(21)) / df['Close'].shift(21)).shift(1)
    df_new['return_365'] = ((df['Close'] - df['Close'].shift(252)) / df['Close'].shift(252)).shift(1)
    df_new['moving_avg_5'] = df_new['return_1'].rolling(5).mean().shift(1)
    df_new['moving_avg_30'] = df_new['return_1'].rolling(21).mean().shift(1)
    df_new['moving_avg_365'] = df_new['return_1'].rolling(252).mean().shift(1)
    # Wartości docelowe
    df_new['close'] = df['Close']
    df_new = df_new.dropna(axis=0)
    return df_new
'''


first_part = '''
# Ustalenie zakresów dat, na które ma być podzielony zbiór danych
start_train = '20-04-1982'
end_train = '31-12-2021'
start_test = '03-01-2022'
end_test = '17-01-2023'

# Utworzenie zbioru treningowego i zbioru testowego
data_train = data.loc[start_train:end_train]
X_train = data_train.drop('close', axis=1).values
y_train = data_train['close'].values
data_test = data.loc[start_test:end_test]
X_test = data_test.drop('close', axis=1).values
y_test = data_test['close'].values


# Znormalizowanie cech, aby miały tę samą lub porównywalną skalę. 
# Polega to na odjęciu od nich średniej wartości i przeskalowaniu ich do jednostkowej wariancji.
# Do przeskalowania zbiorów użyto przetrenowanego obiektu scaler
scaler = StandardScaler()
X_scaled_train = scaler.fit_transform(X_train)
X_scaled_test = scaler.transform(X_test)

# Ustalenie seedu w celu testowania
tf.random.set_seed(42)


# Zbudowanie sieci neuronowej, wykorzystującą klasę Sequential zawartą w bibliotece Keras. 
# Początkowa sieć składa się z jednej warstwy ukrytej, 
# zbudowanej z 32 węzłów i wykorzystującej funkcję aktywacji ReLU
model = Sequential([
    Dense(units=32, activation='relu'),
    Dense(units=1)
])

# Skomplilowanie modelu sieci, wykorzystując optymalizator Adam. 
# Przyjęta szybkość uczenia się 0.1 i błąd średniokwadratowy jako cel treningu.
model.compile(loss='mean_squared_error',
              optimizer=tf.keras.optimizers.Adam(0.1))
model.fit(X_scaled_train, y_train, epochs=100, verbose=True)
predictions = model.predict(X_scaled_test)[:, 0]

# Użycie przetrenowanego modelu do przetworzenia zbioru testowego i wyświetlenie wskaźników skuteczności
print(f'Błąd średniokwadratowy: {mean_squared_error(y_test, predictions):.3f}')
print(f'Średni błąd bezwzględny: {mean_absolute_error(y_test, predictions):.3f}')
print(f'R^2: {r2_score(y_test, predictions):.3f}')
'''


tuning_network_parameters = '''
# Dostosowywana będzie liczba węzłów w ukrytej warstwie, liczbę warstw ukrytych plus wyjściowa, liczba iteracji treningowych i szybkość uczenia.
# Będą to trzy liczby węzłów(wartości dyskretne) równe 13, 32 i 64,
# trzy liczby warstw ukrytych plus wyjściowa równe 2, 3, 5 
# dwie liczby iteracji(wartości dyskretne) równe 300 i 1000 oraz 
# szybkość uczenia(wartość ciągła) z zakresu od 0,01 do 4

HP_HIDDEN = hp.HParam('hidden_size', hp.Discrete([64, 32, 16]))
HP_LAYERS = hp.HParam('layers', hp.Discrete([2, 3, 5]))
HP_EPOCHS = hp.HParam('epochs', hp.Discrete([300, 1000]))
HP_LEARNING_RATE = hp.HParam('learning_rate', hp.RealInterval(0.01, 0.4))


def train_test_model(hparams: Dict[hp.HParam, Any], logdir: str) -> tuple:
    """
    Funkcja trenująca i testująca model sieci neuronowej
    @param hparams: obiekt słownika zawierający hiperparametry w postaci {nazwa_hiperparametru:wartość}
    @param logdir: ścieżka dla każdej iteracji trenowania i testowania sieci
    @return: mse, r2 - obiekt tuple zawierający wskażniki skuteczności modelu    
    """
    # Założono trenowanie i testowanie sieci neuronowej o 3 architekturach:
    #   - jedna warstwa ukryta
    #   - dwie warstwy ukryte, druga z połową węzłów
    #   - dwie warstwy ukryte, druga z połową węzłów, dodatkowo zostosowamy Dropout o wartości 0.5
    if hparams[HP_LAYERS] == 2:
        layers_list = [Dense(units=hparams[HP_HIDDEN], activation='relu'), Dense(units=1)]
    elif hparams[HP_LAYERS] == 3:
        layers_list = [Dense(units=hparams[HP_HIDDEN], activation='relu'), Dense(units=(hparams[HP_HIDDEN]//2), activation='relu'), Dense(units=1)]
    elif hparams[HP_LAYERS] == 5:
        layers_list = [Dense(units=hparams[HP_HIDDEN], activation='relu'), Dropout(0.5), Dense(units=(hparams[HP_HIDDEN]//2), activation='relu'), Dropout(0.5), Dense(units=1)]

    model = Sequential(layers_list)
    model.compile(loss='mean_squared_error',
                  optimizer=tf.keras.optimizers.Adam(hparams[HP_LEARNING_RATE]),
                  metrics=['mean_squared_error'])
    model.fit(X_scaled_train, y_train, validation_data=(X_scaled_test, y_test), epochs=hparams[HP_EPOCHS], verbose=False,
              callbacks=[
                  tf.keras.callbacks.TensorBoard(logdir),
                  hp.KerasCallback(logdir, hparams),
                  tf.keras.callbacks.EarlyStopping(
                      monitor='val_loss', min_delta=0, patience=200, verbose=0, mode='auto',
                  )
              ],
              )
    _, mse = model.evaluate(X_scaled_test, y_test)
    pred = model.predict(X_scaled_test)
    r2 = r2_score(y_test, pred)
    return mse, r2


def run(hparams: Dict[hp.HParam, Any], logdir: str) -> None:
    """
    Funkcja inicjująca proces treningu za pomocą kombinacji hiperparametrów oraz wyświetlająca podsumowanie 
    zawierające wartości błędu średniokwadratowego i współczynnika R^2
    @param hparams: obiekt słownika zawierający hiperparametry w postaci {nazwa_hiperparametru:wartość}
    @param logdir: ścieżka dla każdej iteracji trenowania i testowania sieci   
    """
    with tf.summary.create_file_writer(logdir).as_default():
        hp.hparams_config(
            hparams=[HP_HIDDEN, HP_LAYERS, HP_EPOCHS, HP_LEARNING_RATE],
            metrics=[hp.Metric('mean_squared_error', display_name='mse'),
                     hp.Metric('r2', display_name='r2'),
                    ],
        )
        mse, r2 = train_test_model(hparams, logdir)
        tf.summary.scalar('mean_squared_error', mse, step=1)
        tf.summary.scalar('r2', r2, step=1)



# Tworzenie, kompilowanie i trenowanie modelu sieci neuronowej z wszystkimi możliwymi kombinacjami hiperparametrów.
# W każdej próbie są stosowane trzy wartości dyskretne(liczbę węzłów w warstwie ukrytej, liczbę warstw i liczbę iteracji) wybrane ze zdefiniowanych pól 
# oraz jedna wartość ciągła(szybkość uczenia) wybrana z jednego z równych podprzedziałów z założonego zakresu
session_num = 0
for hidden in HP_HIDDEN.domain.values:
    for layers_number in HP_LAYERS.domain.values:
        for epochs in HP_EPOCHS.domain.values:
            for learning_rate in tf.linspace(HP_LEARNING_RATE.domain.min_value, HP_LEARNING_RATE.domain.max_value, 5):
                hparams = {
                    HP_HIDDEN: hidden,
                    HP_LAYERS: layers_number,
                    HP_EPOCHS: epochs,
                    HP_LEARNING_RATE: float("%.2f"%float(learning_rate)),
                }
                # W chwili rozpoczęcia prób tworzone są katalogi, 
                # w których zapisywane są wyniki treningu i weryfikacji modelu w każdej próbie.
                run_name = "run-%d" % session_num
                print('--- Próba: %s' % run_name)
                print({h.name: hparams[h] for h in hparams})
                run(hparams, 'logs/hparam_tuning/' + run_name)
                session_num += 1
'''
 
sumup1 = '''
Po przeprowadzeniu wszystkich prób można otworzyć panel Tensorboard za pomocą polecenia:
`tensorboard --logdir logs/hparam_tuning`
W zakładce HPARAMS możemy zobaczyć tabele zawierającą wszystkie kombinacje hiperparametrów i opowiadające im wskaźniki skuteczności:
'''

sumup2 = '''Jak widać najlepszą skuteczność model osiąga dla kombinacji hiperparametrów layers=3, hidden_size=64, 
epochs=1000 i learning_rate=0.4, dla której współczynnik R^2 równy jest 0.955'''


if __name__== '__main__':
    st.set_page_config(layout="wide", page_title="Stock prices Neural Network")

    with st.sidebar:
        selected = option_menu(
            menu_title=None,
            options=["Description", "Script", "Predictor"],
            icons=["book", "bar-chart", "question"],
            menu_icon='cast',
            default_index=0
        )


    description = rewrite_markdown_script("czesc_opisowa.md")
    gspc = yf.Ticker("^GSPC")
    data_raw_full = gspc.history(period="max")
    data_raw_full = data_raw_full[["Open", "High", "Low", "Close", "Volume"]]
    data_raw = get_dataset()
    prepared_data = data_preparation(data_raw)

    if selected == "Description":
        with st.container():
            st.markdown(description, unsafe_allow_html=True)

    if selected == "Script":
        st.markdown("# Projekt sieci neuronowej przewidującej ceny akcji - część techniczna")
        st.markdown("## Eksploaracyjna analiza danych")

        st.write("S&P 500 dataset")
        st.table(data_raw_full.head(10))

        st.markdown("Wyświetlenie statystyki opisowej")
        st.write(data_raw_full.describe())

        st.markdown("Wyświetlenie ilośći brakujących wartośći")
        st.write(data_raw_full.isna().sum())
        st.markdown("Nie ma takich wartości, nie trzeba uzupełniać")

        st.markdown("Ilość wierszy, w których wartości są większe od zera")
        st.write(data_raw_full[data_raw_full > 0].count())
        st.markdown("Jak widać w kolumnach Open i Volume znajdują się wiersze z wartościami 0, czyli są niekompletne")
        st.markdown("W takim przypadku bierzemy zakres danych po 1982-04-20, gdzie wszystkie wiersze mają wartości większe od zera")

        st.markdown("Jak widać wszsytkie wartości są większe od zera")
        st.write(data_raw[data_raw > 0].count())

        plot_data(data_raw)

        st.markdown("## Projektowanie sieci neuronowej")
        st.markdown("### Trening podstawowej sieci neuronowej")
        st.markdown("#### Funkcja tworząca cechy")
        st.code(generate_features_function, language='python')
        st.markdown("#### Przygotowanie danych i funkcja trenująca")
        st.code(first_part, language='python')

        st.markdown("#### Wyniki trenowania początkowego modelu")
        train_default_NN(prepared_data)
        st.markdown("### Dostrojenie parametrów sieci neuronowej za pomocą modułu hparams")
        st.code(tuning_network_parameters, language='python')

        st.markdown(sumup1)
        st.image("./data/TensorBoard_hparams_top_r2.png")
        st.markdown(sumup2)

        st.markdown("### Użycie optymalnego modelu do wyliczenia prognoz")

        final_predictions = final_neural_network(prepared_data)
        predictions_plot(final_predictions, prepared_data)
    
    if selected == "Predictor":
        st.markdown("## Przywidywana cena indeksu w następnym dniu")
        price = predict_next_day_price(prepared_data.X_train)
        st.markdown(f"Przewidywana cena: {price:.2f}$")