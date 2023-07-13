import streamlit as st
import numpy as np
import pandas as pd
import sklearn
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier

# backEnd

data = pd.read_csv("phone.csv")
df = data.drop(['clock_speed', 'm_dep', 'talk_time', 'wifi'], axis=1)
df['sc_size'] = (((df['sc_h']) * (df['sc_h'])) + ((df['sc_w']) * (df['sc_w']))) ** 0.5 * 0.393701

df.drop(['sc_h', 'sc_w'], axis=1, inplace=True)

x = df.drop(['price_range'], axis=1)
y = df['price_range']

# sc = StandardScaler()
# x_scaled  = sc.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

rfc = RandomForestClassifier(criterion='entropy', random_state=42)
rfc.fit(x_train, y_train)
rfc_pred_test = rfc.predict(x_test)

from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, rfc_pred_test)

# FrontEnd

st.title(" Let's Predict your ideal phone price range!")
st.image("Mobile-Phone-PNG-File.jpg")
st.header('please enter phone\'s information')


def user_input():
    st.subheader('**screen and battery**')
    BatteryPower = st.slider('Battery Capacity(mAh)', 200, 6000)
    touchScreen = 0
    if st.checkbox('Touch screen'):
        touchScreen = 1
    scSize = st.slider('Screen Size(inch)', 2.0, 7.0)
    pxHieght = st.slider('Pixel Height', 1, 3000)
    pxwidth = st.slider('Pixel width', 1, 3000)
    mobileWt = st.slider('phone weight(grams)', 50, 500)
    st.subheader('**network**')
    blue = st.checkbox('Bluetooth')
    sim = st.radio('Number of sim', [1, 2])
    sim1 = 0
    sim2 = 0
    if sim == 1:
        sim1 = 1
    elif sim == 2:
        sim2 = 1

    threeG = 0
    if st.checkbox('3G'):
        threeG = 1
    fourG = 0
    if st.checkbox('4G'):
        fourG = 1

    st.subheader('**Camera**')
    pc = st.slider(' primary camera ', 1, 50)
    fc = st.slider(' Front camera ', 1, 50)

    st.subheader('**Platform**')
    intMemory = st.radio(' Internal Memory ', [1, 2, 4, 8, 16, 32, 64, 128, 256])
    ram = st.radio(' RAM ', [1, 2, 4, 8, 16, 32])
    nCore = st.radio(' Number of cores ', [1, 2, 3, 4, 5, 6, 7, 8])
    dt = {'battery_power': int(BatteryPower),
          'blue': int(blue),
          "dual_sim": int(sim2),
          'fc': int(fc),
          'four_g': int(fourG),
          'int_memory': int(intMemory),
          'mobile_wt': int(mobileWt),
          'n_cores': int(nCore),
          'pc': int(pc),
          'px_height': int(pxHieght),
          'px_width': int(pxwidth),
          'ram': int(ram) * 1024,
          'three_g': int(threeG),
          'touch_screen': int(touchScreen),
          'sc_size': int(scSize)}
    feat = pd.DataFrame(dt, index=[0])
    return feat


d = user_input()
# d = sc.fit_transform(d)
predection = rfc.predict(d)
st.header('prediction')
if predection == 0:
    st.write('The phone probably is under 200$')
elif predection == 1:
    st.write('The phone probably is between 200$ and 500$')
elif predection == 2:
    st.write('The phone probably is between 500$ and 700$')
elif predection == 3:
    st.write('The phone probably is more than 1000$')
