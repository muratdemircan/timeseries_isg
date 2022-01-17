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


    st.markdown("Dataseti aylık olarak yeniden oluşturduk")
