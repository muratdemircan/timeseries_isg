import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from keras import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras import layers
from keras import Input
import math

# streamlit run isg_date_streamlit.py

def get_first_time(col):
    t = ""
    try:
        t = col.split("-")[0].strip() + ":00"
    except Exception as e:

        print(t, "{}, {} hatasi".format(t, str(e)))
        t = "00:00:00"
    return t

def process_scalar(month_df):
    dataset = month_df.values
    dataset = dataset.astype("float32")
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    return dataset, scaler


rename_columns = {
    "Proje": "proje",
    "Tarih": "tarih",
    "Ana Faaliyet Alanı": "faaliyet_alan",
    "Kök Sebep": "sebep",
    "Olay Tipi": "tip",
    "Yüklenici_Altyüklenici": "yuklenici",
    "Kişisel Etkenler": "etkenler",
    "Görev": "gorev",
    "Lokasyon": "konum",
    "Departman": "departman",
    "Yaşı": "yas",
    "Projede Çalışma Süresi": "working_day",
    "İl": "il",
    "Gündüz Sıcaklığı": "temp",
    "Hava Durumu": "weather",
    "Projenin Resmi bitişine kalan süre": "last_project_days",
    "Targets": "targets"

}

def create_montly_table(df, rename_columns):
    # df.drop(["No", "Proje-No", "Proje-Ay-Yıl", "İl-Kaza Tarihi"], axis=1, inplace=True)
    df["Saat"] = df["Saat"].apply(get_first_time)
    df["Tarih"] = pd.to_datetime(df["Tarih"])
    df["Tarih"] = df["Tarih"].dt.strftime('%d-%m-%Y')
    df.rename(columns=rename_columns, inplace=True)
    df['date'] = pd.to_datetime(df.tarih) - pd.offsets.MonthEnd(0) - pd.offsets.MonthBegin(1)
    df = df[["date"]]
    df = df["date"].value_counts().sort_index()
    month_df = df.to_frame()
    month_df = month_df.reset_index()
    month_df.rename({"index": "month", "date": "value"}, axis=1, inplace=True)
    month_df.set_index("month", inplace=True)
    return month_df

def forecast(model, last, n):
    in_value = last.copy()
    preds = []
    for i in range(n):
        p = model.predict(in_value)
        preds.append(p)
        in_value = np.append(in_value, p)[1:].reshape(last.shape)
    return np.array(preds).flatten()

def to_sequence(dataset, seq_size=1):
    x = []
    y = []
    for i in range(len(dataset) - seq_size - 1):
        window = dataset[i:(i + seq_size), 0]
        x.append(window)
        y.append(dataset[i + seq_size, 0])

    return np.array(x), np.array(y)

def getLag(series, n):
    X, y = [], []
    for i in range(len(series) - n -1):
        X.append(series[i : (i+n), 0])
        y.append(series[i+n, 0])
    return np.array(X), np.array(y), series[-n:].reshape(1,n)

def model_visualize(history):
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.title("Validation Loss Curve MSE")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()


def create_model(seq_size, scaler):
    model = Sequential()
    model.add(Dense(64, input_dim=seq_size, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error", optimizer="adam", metrics=["acc", "mse", "accuracy"])
    # print(model.summary())
    history = model.fit(trainX, trainY, validation_data=(testX, testY), verbose=2, epochs=100)

    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    trainPredict = scaler.inverse_transform(trainPredict)
    testPredict = scaler.inverse_transform(testPredict)

    trainY_inverse = scaler.inverse_transform(trainY.reshape(1, -1))
    testY_inverse = scaler.inverse_transform(testY.reshape(1, -1))

    trainScore = math.sqrt(mean_squared_error(trainY_inverse[0], trainPredict[:, 0]))
    testScore = math.sqrt(mean_squared_error(testY_inverse[0], testPredict[:, 0]))
    print("Train Score: {} RMSE".format(trainScore))
    print("Test Score: {} RMSE".format(testScore))
    return model, trainPredict, testPredict


def visualize_results(scaler, trainPredict, testPredict, month_df):
    # shift train predictions for plotting
    # we must shift the predictions so that they align on the x-axis with the original dataset.
    trainPredictPlot = np.empty_like(dataset)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[seq_size:len(trainPredict) + seq_size, :] = trainPredict

    # shift test predictions for plotting
    testPredictPlot = np.empty_like(dataset)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict) + (seq_size * 2) + 1:len(dataset) - 1, :] = testPredict


    # Whole dataframe
    st.write("Yıllar içerisinde gerçekleşene İSG Aylık Kaza Sayıları")
    dt = pd.DataFrame(scaler.inverse_transform(dataset),index = month_df.index, columns = ["Kayıtlı Kaza"])
    # st.write(dt.head())
    st.line_chart(dt["Kayıtlı Kaza"])

    # Train prediction dataframe
    train_pre_df = pd.DataFrame(trainPredictPlot, index = month_df.index, columns = ["Train"])
    # st.write(train_pre_df.head())
    # st.line_chart(train_pre_df["values"])

    # Test prediction dataframe
    test_pre_df = pd.DataFrame(testPredictPlot, index=month_df.index, columns=["Test"])
    # st.write(test_pre_df.head())
    # st.line_chart(test_pre_df["values"])

    st.write("İSG Aylık Kaza Sayılarının çalışıldığı zaman bazlı training ve test datasetlerinin görüntülenmesi")

    predictions = pd.concat([dt, train_pre_df, test_pre_df], axis = 1)
    # st.write(predictions.head())
    st.line_chart(predictions)

    return predictions



    # st.line_chart(trainPredictPlot)
    # st.line_chart(testPredictPlot)

    # plt.grid()
    # plt.legend()
    # plt.show()


def visualize_forecast(forecasted_df, dataset, model_test):
    # shift test predictions for plotting
    predictplot = np.empty_like(dataset)
    predictplot[:, :] = np.nan

    # preds_dataset = np.append(predictplot, preds)
    # print(preds_dataset.shape)
    # print(month_df.shape)
    # forecast_df = pd.DataFrame(preds_dataset, index=time_index, columns=["Forecasted"])

    predictions = pd.concat([model_test, forecasted_df], axis=1)
    # st.write(predictions.head())
    st.line_chart(predictions)

    # plt.plot(scaler.inverse_transform(dataset), color='blue')
    # plt.plot(scaler.inverse_transform(preds.reshape(-1, 1)), color='red')
    # plt.legend(loc='Left corner')



header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()
visualize_model = st.container()

with header:
    st.title("İSG Kaza Tahminleme Projesi")
    st.text("Bu projede kaza sayılarının tahminlemesi gerçekleştirilecektir")

with dataset:
    st.header("Kaza Kayıtları dataseti")
    st.text("Örnek dataset")

    #Dataset alınırken hepsi string olarak alındı. Streamlit 1.3 versiyonundan kaynaklı bir numpy kütüphanesinden hata alınıyordu

    df = pd.read_excel(r"data\kaza_date.xlsx").astype(str)
    st.write(df.head())

    st.markdown("Dataseti aylık olarak yeniden oluşturduk")

    month_df = create_montly_table(df, rename_columns)
    arr = np.random.normal(1, 1, size=100)
    # fig, ax = plt.subplots()
    # ax.plot(month_df["value"])
    # st.pyplot(fig)

    # Zaman bazlı toplam kaza sayılarının görüntülenmesi
    st.line_chart(month_df["value"])




with model_training:
    st.header("Time to train the model")
    # Scalar process
    dataset, scaler = process_scalar(month_df)

    # Train, test splitting
    train_size = int(len(dataset) * 0.66)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

    seq_size = 7

    # trainX, trainY = to_sequence(train, seq_size)
    # testX, testY = to_sequence(test, seq_size)

    trainX, trainY, last_train = getLag(train, seq_size)
    testX, testY, last_test = getLag(test, seq_size)




with visualize_model:
    model, trainPredict, testPredict = create_model(seq_size, scaler)
    st.write("Aylık İSG kazalarının sayısı alınmıştır. ML model çalışmasında önceki 7 ay kaza sayıları dikkate alınarak model çalışması gereçekleştirilmiştir.")
    model_test = visualize_results(scaler, trainPredict, testPredict, month_df)

    n_month_for_forecast = 24

    predict_period_dates = pd.date_range(month_df.index[-1], periods=n_month_for_forecast, freq='MS').tolist()
    predict_period_dates = predict_period_dates
    # print(predict_period_dates)

    preds = forecast(model, last_test, n_month_for_forecast)
    # forecasted = scaler.inverse_transform(preds)
    # forecasted = scaler.inverse_transform(preds)
    forecasted = scaler.inverse_transform(preds.reshape(-1, 1))
    print(forecasted.shape)
    print(len(predict_period_dates))
    forecasted_df = pd.DataFrame(forecasted, index = predict_period_dates, columns=["Forecasted"])
    st.write(forecasted_df.head())
    # vis_dates = list(month_df.index) + predict_period_dates

    visualize_forecast(forecasted_df, dataset, model_test)