import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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
