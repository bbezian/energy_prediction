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



# Title of the app
st.title('Upload CSV File')

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
    selected_row = st.text_input("Select a date")
    # Display the selected row
    #st.write(df.iloc[selected_row])
    selected_row = df[df['date']==str(selected_row)]
    st.write(selected_row)


    # Allow the user to select a column from the selected row
    selected_column = st.selectbox("Select a column", df.columns)

    # Display the selected column value from the selected row
    selected_value = selected_row[selected_column].iloc[0]
    st.write(f"{selected_column}: {selected_value}")



enthalpy = selected_row['enthalpy'].iloc[0]    
X = enthalpy
X_tst = load_scaler.transform([[X]])

energy_predicted = load_model.predict(X_tst)

energy_predicted = energy_predicted.reshape(len(energy_predicted),1)
energy_predicted = load_scaler_y.inverse_transform(energy_predicted)
energy_predicted = energy_predicted.reshape(len(energy_predicted),1)

st.subheader(f'The predicted energy consumption is : {float(energy_predicted):.4f}')

measured_energy_consumption = selected_row['energy'].iloc[0]  
st.subheader(f'Measured energy consumption : {measured_energy_consumption: .4f}')


Threshold = 306.62016807
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
   



