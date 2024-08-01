# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 18:46:38 2023

@author: abhij
"""

import numpy as np
import pickle

# loading the saved model
loaded_model = pickle.load(open("C:/Users/ASHWATHI/Desktop/alzheimers/trained_model.sav", 'rb'))


input_data = (1,80,16,2,29,0,1323,0.738,1.326)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('Demented')
else:
  print('Non Demented')