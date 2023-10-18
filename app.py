import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
data = pd.read_csv('in-vehicle-coupon-recommendation.csv')

# Defining the functions for each page
def home():
   st.title('Home Page')
   #st.image('img1.jpeg', use_column_width=True)
   st.write('Welcome to Streamlit app!')
   st.write(data.head())
   
  

def eda():
   st.title('EDA Page')
   # ...

def prediction():
   st.title('Prediction Page')
   # ...

# Use streamlit to create a multi-page app with a navigation sidebar
st.sidebar.title('Navigation')
options = ['Home', 'EDA', 'Prediction']
selection = st.sidebar.selectbox('Select a page', options)

# Add the necessary code to display the appropriate page based on the user's selection
if selection == 'Home':
   home()
elif selection == 'EDA':
   eda()
else:
   prediction()
