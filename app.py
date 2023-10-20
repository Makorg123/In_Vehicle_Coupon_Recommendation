import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import altair as alt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# page config
st.set_page_config(
    page_title="CruizeSaver", 
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
# data = pd.read_csv('in-vehicle-coupon-recommendation.csv')
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- # 
# Defining the functions for each page
def home():
    st.header('Welcome to :orange[CruizeSaver]: *Your* :violet[Vehicle Coupon Advisor]')
    st.subheader('Unlock Savings, Elevate Your Ride')
    # st.image('https://images.unsplash.com/photo-1581093458791-9d8a5a6a9f1a?ixid=MnwxMjA3fDB8MHxzZWFyY2h8Mnx8Y291cG9uJTIwYXJjaGl2ZXxlbnwwfHwwfHw%3D&ixlib=rb-1.2.1&w=1000&q=80', width=700)
    st.write("**CruizeSaver** is your destination for intelligent savings tailored to your vehicle. We've harnessed the power of data and predictive analytics to optimize your driving experience while keeping more money in your pocket. Welcome to a world where smart savings meet your automotive passion.")
    st.subheader('Your Path to Smart Savings')
    col1,col2 = st.columns(2)
    with col1:
      st.subheader('Discover Data Insights:')
      st.write('Embark on a journey into the world of automotive data with our comprehensive **Exploratory Data Analysis (EDA)**. Our expertly crafted analysis offers a unique perspective on driving trends, consumer preferences, and the wealth of insights that drive our coupon recommendations.')

    with col2:
      st.subheader('Coupon Prediction:')
      st.write("Our **Coupon Prediction** tool is the first of its kind. We've trained a machine learning model to predict the best coupon for your vehicle based on your driving habits. Simply enter your vehicle information and driving preferences to unlock your savings today!")

    st.subheader('Your Savings, Your Way, Your CruizeSaver')
    st.write("We're excited to have you join us on this journey. We're confident that you'll find our platform to be a valuable resource for your automotive needs. We're always looking for ways to improve, so please don't hesitate to reach out with any feedback or suggestions. We look forward to hearing from you! *Get Started*")
    st.write("Sincerely,")
    st.subheader("**Mohammed Anas khan**")
    linkedin_url = "https://www.linkedin.com/in/mohammed-anas-khan-ab91531a4/"
    github_url = "https://github.com/Makorg123"

    # Add links to your LinkedIn and GitHub profiles
    st.write(f"LinkedIn: [My LinkedIn Profile]({linkedin_url})", f"GitHub: [My GitHub Profile]({github_url})")
  
# ------------------------------------------------------------------------------------------------------------ #

def eda():
    data = pd.read_csv('in-vehicle-coupon-recommendation.csv')
    st.header('Exploratory Data Analysis (EDA): Uncover Insights for Smarter Savings')
    st.write('In our EDA section, we take a deep dive into the world of automotive data. Discover trends, patterns, and valuable insights that drive our intelligent vehicle coupon recommendations. We dissect a wealth of information to understand your driving habits, vehicle preferences, and consumer trends. Our EDA not only informs our recommendations but also empowers you with a unique perspective on the road ahead. Join us in exploring the data that makes your drive smarter and your savings greater.')

    # cleaning the data
    data.drop(['car','toCoupon_GEQ5min'],axis = 1)

    # mode imputation for missing values in data
    data['Bar']=data['Bar'].fillna(data['Bar'].value_counts().index[0])
    data['CoffeeHouse']=data['CoffeeHouse'].fillna(data['CoffeeHouse'].value_counts().index[0])
    data['CarryAway']=data['CarryAway'].fillna(data['CarryAway'].value_counts().index[0])
    data['RestaurantLessThan20']=data['RestaurantLessThan20'].fillna(data['RestaurantLessThan20'].value_counts().index[0])
    data['Restaurant20To50']=data['Restaurant20To50'].fillna(data['Restaurant20To50'].value_counts().index[0])

    # remove duplicates.
    duplicate = data[data.duplicated(keep = 'last')]
    data = data.drop_duplicates()
    st.divider()
    # st.write("Shape of dataset after removing duplicates:",data.shape)

    # pie chart for temperature value counts
    col1,col2,col3 = st.columns(3)
    #with col1:
     # st.subheader('Temp Value Counts')
      #fig = plt.figure(figsize=(5,5))
      #plt.pie(data['temperature'].value_counts(),labels=data['temperature'].value_counts().index,autopct='%1.1f%%')
      #st.pyplot(fig)

    with col1:
      st.subheader('Temp Value Counts')
      explode = (0, 0.1, 0)  
      colors = ['#FFC300', '#FF5733', '#C70039']
      fig1, ax1 = plt.subplots()
      ax1.pie(data['temperature'].value_counts(),explode=explode,labels=data['temperature'].value_counts().index,
      autopct='%2.2f%%',colors=colors,shadow=True, startangle=90)
      ax1.axis('equal') 
      st.pyplot(fig1)
   
    # pie chart for weather value counts
    with col2:
      st.subheader('Weather Value Counts')
      explode = (0.1, 0, 0)  
      colors = ['#FFC300', '#FF5733', '#C70039']
      fig2, ax2 = plt.subplots()
      ax2.pie(data['weather'].value_counts(),explode=explode,labels=data['weather'].value_counts().index,
      autopct='%1.1f%%',colors=colors,shadow=True, startangle=90)
      ax2.axis('equal')
      st.pyplot(fig2)

    # pie chart for time value counts
    with col3:
      st.subheader('Time Value Counts')
      explode = (0.1, 0, 0, 0, 0)  
      colors = ['#FFC300', '#FF5733', '#C70039', '#E3691A', '#F2A487']
      fig3, ax3 = plt.subplots()
      plt.pie(data['time'].value_counts(),explode=explode,labels=data['time'].value_counts().index,
      autopct='%2.1f%%',colors=colors,shadow=True, startangle=90)
      ax3.axis('equal')
      st.pyplot(fig3)
    st.divider()

    col4,col5 = st.columns(2)

    # bar chart for coupon distribution
    with col4:
      st.subheader('Coupon Distribution')
      chart_data = data.coupon.value_counts()
      st.bar_chart(chart_data)

    # gender education distribution
    with col5:
      st.subheader('Gender education distribution')
      ax7 = pd.crosstab(data.gender,data.education).plot(kind='bar')
      for i in ax7.containers:
         ax7.bar_label(i)
      st.pyplot(ax7.figure)
    st.divider()

    # Income Category-wise Accepted Coupons
    pivoted_data = data.pivot_table(index = 'income',columns = 'Y',aggfunc='size',fill_value=0)
    fig7, ax = plt.subplots(figsize=(10,6))
    pivoted_data.plot(kind='barh',stacked=True,ax=ax)
    ax.set_xlabel("Number of Coupons")
    ax.set_ylabel("Income Category")
    ax.set_title("Income Category-wise Accepted Coupons")
    for p in ax.patches:
      width, height = p.get_width(), p.get_height()
      x,y = p.get_xy()
      ax.annotate(f'{width}', (x+width/2, y+height/2), ha = 'center')
    st.pyplot(fig7)
    
    col6,col7 = st.columns(2)
    # Coupon expiration vs time
    with col6:
      st.subheader('Coupon expiration vs time')
      ax=pd.crosstab(data.time,data.expiration).plot(kind='bar')
      for i in ax.containers:
         ax.bar_label(i)
      st.pyplot(ax.figure)

    with col7:
      st.subheader('Coupon expiration vs destination')
      ax = pd.crosstab(data.destination, data.Y).plot(kind='bar')
      for i in ax.containers:
         ax.bar_label(i)
      st.pyplot(ax.figure)

    st.divider()
    df_y_0 = data[data['Y'] == 0]
    df_y_1 = data[data['Y'] == 1]
    st.subheader('At what time, which coupon acceptance and rejection ratio is high?')

    st.write('Time vs. coupon (y = 0)')
    ax = pd.crosstab(df_y_0.time,df_y_0.coupon).plot(kind='bar',figsize=(15,6))
    for i in ax.containers:
      ax.bar_label(i)
    st.pyplot(ax.figure)

    st.write('Time vs. coupon (y = 1)')
    ax = pd.crosstab(df_y_1.time,df_y_1.coupon).plot(kind='bar',figsize=(15,6))
    for i in ax.containers:
      ax.bar_label(i)
    st.pyplot(ax.figure)
    st.divider()

    st.subheader('Bivariate Analysis of Coupon type and its Expiration.')
    st.write('Coupon type and Expiration (y = 0)')
    ax = pd.crosstab(df_y_0.expiration,df_y_0.coupon).plot(kind='bar',figsize=(15,6))
    for i in ax.containers:
      ax.bar_label(i)
    st.pyplot(ax.figure)

    st.write('Coupon type and Expiration (y = 1)')
    ax = pd.crosstab(df_y_1.expiration,df_y_1.coupon).plot(kind='bar',figsize=(15,6))
    for i in ax.containers:
      ax.bar_label(i)
    st.pyplot(ax.figure)
    


# -------------------------------------------------------------------------------------------------------------- #
def prediction():
   st.subheader('Prediction Page')

   col1,col2,col3,col4,col5,col6 = st.columns(6)
   with col1:
     destination = st.selectbox('Destination',('No Urgent Place','Home','Work'))

     passanger = st.selectbox('Passanger',('Alone','Friend(s)','Partner','Kid(s)'))

     weather = st.selectbox('Weather',('Sunny','Snowy','Rainy'))

     temperature = st.selectbox('Temperature',('80','55','30'))

   with col2:
     time = st.selectbox('Time',('2PM','10AM','6PM','7AM','10PM'))

     coupon = st.selectbox('Coupon',('Restaurant(<20)','Coffee House','Carry out & Take away','Bar','Restaurant(20-50)'))

     expiration = st.selectbox('Expiration',('1d','2h'))

     gender = st.selectbox('Gender',('Male','Female'))
    
   with col3:
      age = st.selectbox('Age',('21','46','26','31','41','50plus','36','below21'))

      maritalStatus = st.selectbox('Marital Status',('Unmarried partner','Single','Married partner','Divorced','Widowed'))

      has_children = st.selectbox('Has Children',('1','0'))

      education = st.selectbox('Education',('Some college - no degree','Bachelors degree','Associates degree','High School Graduate','Graduate degree (Masters or Doctorate)','Some High School'))

   with col4:
      occupation = st.selectbox('Occupation',('Unemployed','Architecture & Engineering',
                                'Student','Education&Training&Library','Healthcare Support',
                                'Sales & Related','Management','Arts Design Entertainment Sports & Media',
                                'Computer & Mathematical','Life Physical Social Science',
                                'Personal Care & Service','Community & Social Services',
                                'Office & Administrative Support','Construction & Extraction','Legal',
                                'Installation Maintenance & Repair','Business & Financial',
                                'Food Preparation & Serving Related','Production Occupations',
                                'Building & Grounds Cleaning & Maintenance','Transportation & Material Moving',
                                'Protective Service','Healthcare Practitioners & Technical',
                                'Farming Fishing & Forestry','Retired','Military'))
  
      income = st.selectbox('Income',('$25000 - $37499','$12500 - $24999','$37500 - $49999','$100000 or More','$50000 - $62499','Less than $12500','$87500 - $99999','$75000 - $87499','$62500 - $74999'))
  
      Bar = st.selectbox('Bar',('never','less1','1~3','gt8','4~8'))
  
      CoffeeHouse = st.selectbox('CoffeeHouse',('never','less1','4~8','1~3','gt8'))

   with col5:
      CarryAway = st.selectbox('CarryAway',('never','less1','1~3','4~8','gt8'))

      RestaurantLessThan20 = st.selectbox('RestaurantLessThan20',('4~8','1~3','less1','never','gt8'))

      Restaurant20To50 = st.selectbox('Restaurant20To50',('1~3','less1','never','4~8','gt8'))

      toCoupon_GEQ15min = st.selectbox('toCoupon_GEQ15min',('1','0'))


   with col6:
      toCoupon_GEQ25min = st.selectbox('toCoupon_GEQ25min',('1','0'))

      direction_same = st.selectbox('direction_same',('0','1'))

      direction_opp = st.selectbox('direction_opp',('0','1'))
   
   if st.button('Predict'):
    # load the model
    import pickle
    model = pickle.load(open('hgb.pkl','rb'))

    # apply model to make prediction
    prediction = model.predict([['destination', 'passanger', 'weather', 'temperature', 'time', 'coupon',
       'expiration', 'gender', 'age', 'maritalStatus', 'has_children',
       'education', 'occupation', 'income','Bar', 'CoffeeHouse',
       'CarryAway', 'RestaurantLessThan20', 'Restaurant20To50','toCoupon_GEQ15min', 
       'toCoupon_GEQ25min','direction_same', 'direction_opp']])

    # display prediction
    st.success(f'Your predicted coupon is {prediction}')
    st.balloons()






# -------------------------------------------------------------------------------------------------------------- # 

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