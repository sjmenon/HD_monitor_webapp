#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

@author: Sneha Menon
"""

import numpy as np
import pickle
import streamlit as st


loaded_model=pickle.load(open('trained_HDmodel.sav','rb'))
#input_data=[4,110,92,0,0,37.6,0.191,30]

def HD_prediction(input_data):
    

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return 'The person has no heart disease'
    else:
      return 'The person has heart disease'
  

def main():
    st.title('HEART DISEASE PREDICTION MONITOR')
  
    age = st.text_input('age')
    sex = st.text_input('sex')
    cp = st.text_input('cp')
    trestbps = st.text_input('trestbps')
    chol = st.text_input('chol')
    fbs = st.text_input('fbs')
    restecg = st.text_input('restecg')
    thalach = st.text_input('thalach')
    exang = st.text_input('exang')
    oldpeak = st.text_input('oldpeak')
    slope = st.text_input('slope')
    ca = st.text_input('ca')
    thal = st.text_input('thal')
    
    
    # code for Prediction
    diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Test Result'):
        diagnosis = HD_prediction([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal])
        
        
    st.success(diagnosis)
    
    
    
    
    
if __name__ == '__main__':
    main()
  

      






        
