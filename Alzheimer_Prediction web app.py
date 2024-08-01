# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 18:47:54 2023

@author: abhij
"""


import numpy as np
import pickle
import streamlit as st


# loading the saved model
loaded_model = pickle.load(open("D:/KELTRON/JAVA/JAVA_PROJECT/alzheimers/trained_model.sav", 'rb'))


# creating a function for Prediction

def alzheimer_prediction(input_data):
    

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print (prediction)

    if (prediction[0] == 0):
      return 'Demented'
    else:
      return 'Non Demented'

  
def main():
    
    
    # giving a title
    st.title('Alzheimer Prediction Web App')
    
    
    # getting the input data from the user
    
    
    M_F = st.text_input('M_F')
    Age = st.text_input('Age')
    EDUC = st.text_input('EDUC')
    SES = st.text_input('SES')
    MMSE = st.text_input('MMSE')
    CDR = st.text_input('CDR')
    eTIV = st.text_input('eTIV')
    nWBV = st.text_input('nWBV')
    ASF = st.text_input('ASF')
    
    
    
    # code for Prediction
    diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Alzheimer Test Result'):
        diagnosis = alzheimer_prediction([M_F,Age,EDUC,SES,MMSE,CDR,eTIV,nWBV,ASF])
        
        
    st.success(diagnosis)
    
    
    
    
    
if __name__ == '__main__':
    main()
    
    
    