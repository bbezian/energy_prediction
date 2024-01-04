import streamlit as st
import os
import joblib 
import numpy as np 
import pandas as pd

load_scaler_file_path = os.path.abspath('scaler.pkl')
load_scaler_y_file_path = os.path.abspath('scaler_y.pkl')
load_model_file_path = os.path.abspath('energy_prediction_model.pkl')

load_scaler = joblib.load(load_scaler_file_path)
load_scaler_y = joblib.load(load_scaler_y_file_path)
load_model = joblib.load(load_model_file_path)

data = pd.read_csv('data.csv')#,index_col=0

st.title('Energy prediction')
st.write('''We need outdoor enthalpy to predict the energy consumption''')


enthalpy = st.number_input("Enter your enthalpy value")
st.subheader(f'Measured enthalpy value : {enthalpy: .4f}')


X = enthalpy
X_tst = load_scaler.transform([[X]])

energy_predicted = load_model.predict(X_tst)

energy_predicted = energy_predicted.reshape(len(energy_predicted),1)
energy_predicted = load_scaler_y.inverse_transform(energy_predicted)
energy_predicted = energy_predicted.reshape(len(energy_predicted),1)

st.subheader(f'The predicted energy consumption is : {float(energy_predicted):.4f}')

measured_energy_consumption = st.number_input('Enter measured energy consumption')
st.subheader(f'Measured energy consumption : {measured_energy_consumption: .4f}')


#Threshold = 306.62016807
Threshold = st.number_input('Enter specified threshold')
st.subheader(f'Threshold for abnormal energy consumption : {Threshold: .4f}')


residual = (measured_energy_consumption - energy_predicted)

if residual > 0:

    if abs(residual) > Threshold:
        st.subheader(f'Is there high abnormal energy consumption : Yes, there is {float(residual): .4f} higher energy consumption')
    else:
        st.subheader(f'Is there abnormal energy consumption : No')


else:
    if abs(residual) > Threshold:
        st.subheader(f'Is there low abnormal energy consumption : Yes, there is {float(residual): .4f} lower energy consumption')
    else:
        st.subheader(f'Is there abnormal energy consumption : No')
   



