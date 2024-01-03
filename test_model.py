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


# Title of the app
st.title('Upload CSV File for the detection of abnormal energy consumption')

# Upload a CSV file
uploaded_file = st.file_uploader("data.csv", type=["csv"])

# Check if a file was uploaded
if uploaded_file is not None:
    # Read the CSV file
    df = pd.read_csv(uploaded_file)

    # Display the DataFrame
    st.write("### Displaying the DataFrame:")
    st.write(df)


    enthalpy = df['enthalpy']   
    X_tst = load_scaler.transform(enthalpy.values.reshape(-1, 1))

    energy_predicted = load_model.predict(X_tst)

    energy_predicted = energy_predicted.reshape(len(energy_predicted),1)
    energy_predicted = load_scaler_y.inverse_transform(energy_predicted)
    energy_predicted = energy_predicted.reshape(len(energy_predicted),1)

    df['energy_predicted'] = energy_predicted 
    df['residual'] = df['energy'] - df['energy_predicted'] 

    Threshold = 306.62016807
    
    df['high_consumption'] = (df['residual'] > Threshold)*1
    df['low_consumption'] = (df['residual'] < -Threshold)*-1

    st.write("### Displaying the predicted energy consumption:")
    st.write(df)

    st.write("### Displaying the detected high energy consumption:")
    st.write(df[df['high_consumption']==1][['date','energy','energy_predicted','residual']])








  