import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# page config
st.set_page_config(
    page_title="in vehicle coupon", 
    page_icon="🎫", 
    layout='wide',
    initial_sidebar_state="collapsed",
    #page_bg_color="#ADD8E6"  # light blue
)

st.markdown("""
    <style type="text/css">
    blockquote {
        margin: 1em 0px 1em -1px;
        padding: 0px 0px 0px 1.2em;
        font-size: 20px;
        border-left: 5px solid rgb(230, 234, 241);
        # background-color: rgb(129, 164, 182);
    }
    blockquote p {
        font-size: 30px;
        color: #FFFFFF;
    }
    [data-testid=stSidebar] {
        background-color: rgb(129, 164, 182);
        color: #FFFFFF;
    }
    [aria-selected="true"] {
         color: #000000;
    }
    </style>
""", unsafe_allow_html=True)

# Load the dataset
data = pd.read_csv('in-vehicle-coupon-recommendation.csv')

# Defining the functions for each page
def home():
    st.title('Home Page')
    st.image('img1.jpeg', use_column_width=True)
    st.write('Welcome to Streamlit app!')

def eda():
    st.title('EDA Page')
    # Add your EDA content here

def prediction():
    st.title('Prediction Page')
    # Add your prediction content here

# Creating the sidebar with 3 options
options = {
    'Home': '🏠',
    'EDA': '📊',
    'Prediction': '🔮'
}

# Display the selected page content
st.sidebar.title('Navigation')
selected_page = st.sidebar.radio("Select a page", list(options.keys()))
if selected_page == 'Home':
    home()
elif selected_page == 'EDA':
    eda()
elif selected_page == 'Prediction':
    prediction()