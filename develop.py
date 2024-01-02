import streamlit as st
import joblib 
import numpy as np 
import pandas as pd

load_scaler = joblib.load('scaler.pkl')
load_scaler_y = joblib.load('scaler_y.pkl')
load_model = joblib.load('energy_prediction_model.pkl')

data = pd.read_csv('data.csv')#,index_col=0


st.title('Energy prediction')
st.write('''We need outdoor enthalpy to predict the energy consumption''')


enthalpy = st.number_input("Enter your enthalpy value")

text_input = st.text_input("Enter your date")




# Title of the app
st.title("Streamlit CSV Reader")

# Upload a CSV file
uploaded_file = st.file_uploader("data.csv", type=["csv"])

# Check if a file was uploaded
if uploaded_file is not None:
    # Read the CSV file
    df = pd.read_csv(uploaded_file)

    # Display the DataFrame
    st.write("### Displaying the DataFrame:")
    st.write(df)


    # Allow the user to select a row
    selected_row = st.number_input("Select a row index", min_value=0, max_value=len(df)-1, value=0, step=1)
    # Display the selected row
    st.write("### Selected Row:")
    st.write(df.iloc[selected_row])


##enthalpy = (data [data['date']==text_input]['enthalpy'])
##st.subheader(enthalpy)


X = enthalpy

X_tst = load_scaler.transform([[X]])

energy_predicted = load_model.predict(X_tst)

energy_predicted = energy_predicted.reshape(len(energy_predicted),1)
energy_predicted = load_scaler_y.inverse_transform(energy_predicted)
energy_predicted = energy_predicted.reshape(len(energy_predicted),1)


st.subheader(f'The predicted energy consumption is : {float(energy_predicted):.4f}')


Threshold = 306.62016807
st.subheader(f'Threshold for abnormal energy consumption : {Threshold: .4f}')


measured_energy_consumption = 5000
st.subheader(f'Measured energy consumption : {measured_energy_consumption: .4f}')


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
   



