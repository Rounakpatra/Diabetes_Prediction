import numpy as np
import pickle
import streamlit as st
import joblib


#loading the saved trained model

model = joblib.load("diabetes_model.joblib")


#making a function to predict

def make_prediction(input_data):

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction=model.predict(input_data_reshaped)

    if(prediction[0]==0):
        return "The Person is Non-Diabetic"
    else:
        return "The Person is Diabetic"


#main function
def main():

    #title of the webpage
    st.title('Diabetes Prediction Web App')

    #getting input from user
    Pregnancy=st.text_input('Number of Pregnancy')
    Glucose=st.text_input('Glucose level')
    Blood_Pressure=st.text_input('Blood Pressure level')
    Skin_Thickness=st.text_input('Skin Thickness level')
    Insulin=st.text_input('Insulin level')
    BMI=st.text_input('BMI Value')
    Diabetes_Pedigree_Function=st.text_input('DiabetesPedigreeFunction Value')
    Age=st.text_input('Age')

    diagnosis=''

    #creating button for making prediction
    if st.button('Diabetes Test Result'):
        diagnosis=make_prediction([Pregnancy,Glucose,Blood_Pressure,Skin_Thickness,Insulin,BMI,Diabetes_Pedigree_Function,Age])


    st.success(diagnosis)

if __name__=='__main__':
    main()
