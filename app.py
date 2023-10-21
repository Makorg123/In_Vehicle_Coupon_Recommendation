import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# page config
st.set_page_config(
    page_title="CouponCraft", 
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
    st.subheader('Welcome to :blue[CouponCraft:] Your Personalized Coupon Experience!')
    st.write("Discover the power of coupons in enhancing your business. Coupons have proven to be effective tools for marketing products and services while encouraging customer engagement. They create a win-win situation for both companies and customers, making it a vital strategy to boost brand impact and customer loyalty.")

    st.write("However, choosing the :orange[right coupon] for each customer can be a complex task. Every customer :orange[profile responds differently], and offering the :red[wrong coupons] can deter them from your business. The solution? Machine learning techniques! Our app utilizes cutting-edge technology to provide tailored coupon recommendations, ensuring that your customers become frequent visitors and your brand's impact is maximized. Say goodbye to ineffective coupon strategies and hello to a smarter way of engaging your audience!")

    st.subheader(':blue[Your Path to Smart Savings]')
    col1,col2 = st.columns(2)
    with col1:
      st.subheader(':orange[Discover Data Insights:]')
      st.write('Unlock the potential of data with our EDA page, where you can dig deep into your customer data, uncover patterns, and gain valuable insights to inform your business decisions. Visualize your data, identify trends, and make data-driven choices that will drive your success.')

    with col2:
      st.subheader(':green[Coupon Prediction:]')
      st.write("On top of EDA, our app provides a dedicated prediction page, enabling you to foresee customer behavior and anticipate their needs. Harness the power of predictive analytics to stay ahead of the game, tailor your marketing strategies, and ensure that your customers receive the offers and promotions they desire.")

    st.write("Sincerely,")
    st.subheader("Mohammed Anas khan")
    linkedin_url = "https://www.linkedin.com/in/mohammed-anas-khan-ab91531a4/"
    github_url = "https://github.com/Makorg123"

    # Add links to your LinkedIn and GitHub profiles
    st.write(f"LinkedIn: [My LinkedIn Profile]({linkedin_url})", f"GitHub: [My GitHub Profile]({github_url})")
  
# ------------------------------------------------------------------------------------------------------------ #

def eda():
    data = pd.read_csv('in-vehicle-coupon-recommendation.csv')
    st.subheader('Exploratory Data Analysis (EDA): Uncover Insights for Smarter Savings')
    st.write('In our EDA section, we take a deep dive into the world of automotive data. Discover trends, patterns, and valuable insights that drive our intelligent vehicle coupon recommendations. We dissect a wealth of information to understand your driving habits, vehicle preferences, and consumer trends. Our EDA not only informs our recommendations but also empowers you with a unique perspective on the road ahead. Join us in exploring the data that makes your drive smarter and your savings greater.')

    # Description of the dataset
    with st.expander("**Description of the dataset** :orange[**User attributes**]"):
      st.write(""":orange[1.User attributes:]
      \n 1. Gender: Female, Male
      2. Age: 21, 46, 26, 31, 41, 50plus, 36, below21
      3. Marital Status: Unmarried partner, Single, Married partner, Divorced, Widowed
      4. has_Children: 1: has children, 0: No children
      5. Education: Some college — no degree, Bachelors degree, Associates degree, High School Graduate, Graduate degree (Masters or Doctorate), Some High School
      6. Occupation: unique 25 number of occupation of users (Unemployed, Architecture & Engineering, Student,Education&Training&Library, Healthcare Support,Healthcare Practitioners & Technical, Sales & Related, Management,Arts Design Entertainment Sports & Media, Computer & Mathematical,Life Physical Social Science, Personal Care & Service, Community & Social Services, Office & Administrative Support, Construction & Extraction, Legal, Retired, Installation Maintenance & Repair, Transportation & Material Moving, Business & Financial, Protective Service, Food Preparation & Serving Related, Production Occupations, Building & Grounds Cleaning & Maintenance, Farming Fishing & Forestry)
      7. Income: income of user (Less than $12500,$12500 — $24999,$25000 — $37499,$37500 — $49999,$50000 — $62499,$62500 — $74999,$75000 — $87499,$87500 — $99999,$100000 or More)
      8. Car : Description of vehicle which driven by user (Scooter and motorcycle, crossover, Mazda5) (99% of values are missing in this feature)
      9. Bar: how many times does the user go to a bar every month? (never, less1, 1~3, 4~8, gt8, nan)
     10. CoffeeHouse: how many times does the user go to a coffeehouse every month? (never, less1, 1~3, 4~8, gt8, nan)
      11. CarryAway: how many times does the user get take-away food every month? (never, less1, 1~3, 4~8, gt8, nan)
      12. RestaurantLessThan20: how many times does the user go to a restaurant with an average expense per person of less than $20 every month? (never, less1, 1~3, 4~8, gt8, nan)
      13. Restaurant20To50: how many times does the user go to a restaurant with average expense per person of $20 — $50 every month? (never, less1, 1~3, 4~8, gt8, nan)""")

    with st.expander("**Description of the dataset** :orange[**Contextual attributes**]"):
      st.write("""1. Destination: destination of user (No Urgent Place, Home, Work)
      2. Passenger: who are the passengers in the car (Alone, Friend(s), Kid(s), Partner)
      3. Weather: weather when user is driving (Sunny, Rainy, Snowy)
      4. Temperature: temperature in Fahrenheit when user is driving (55, 80, 30)
      5. Time: time when user driving (2PM, 10AM, 6PM, 7AM, 10PM)
      6. toCoupon_GEQ5min: driving distance to the restaurant/cafe/bar for using the coupon is greater than 5 minutes (0,1)
      7. toCoupon_GEQ15min: driving distance to the restaurant/cafe/bar for using the coupon is greater than 15 minutes (0,1)
      8. toCoupon_GEQ25min: driving distance to the restaurant/cafe/bar for using the coupon is greater than 25 minutes (0,1)
      9. direction_same: whether the restaurant/cafe/bar is in the same direction as user’s current destination (0,1)
      10. direction_opp: whether the restaurant/cafe/bar is in the opposite direction as user’s current destination (0,1)""")\

    with st.expander("**Description of the dataset** :orange[**Coupon attributes**]"):
      st.write("""\n1. Coupon: coupon type offered by company (Restaurant(<$20), Coffee House, Carry out & Take away, Bar, Restaurant($20-$50))
                Here, <$20 is the average pay per user in a not too expensive restaurant, Restaurant($20-$50) means the average pay per user is between $20 to $50 which little bit expensive restaurant.
                \n2. Expiration: coupon expires in 1 day or in 2 hours (1d, 2h)""")

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
    with col1:
      # Destination wise accepted coupons
      st.write('**Destination wise accepted coupons**')
      ax = pd.crosstab(data.destination,data.Y).plot(kind='bar')
      for i in ax.containers:
         ax.bar_label(i)
      st.pyplot(ax.figure)

    # pie chart for weather value counts
    with col2:
      # passenger wise accepted coupons
      st.write('**Passenger wise accepted coupons**')
      ax = pd.crosstab(data.passanger,data.Y).plot(kind='bar')
      for i in ax.containers:
        ax.bar_label(i)
      st.pyplot(ax.figure)

    # pie chart for time value counts
    with col3:
    # weather wise accepted coupons
      st.write('**Weather wise accepted coupons**')
      ax = pd.crosstab(data.weather,data.Y).plot(kind='bar')
      for i in ax.containers:
       ax.bar_label(i)
      st.pyplot(ax.figure)
    st.divider()

    col4,col5,column6 = st.columns(3)

    # bar chart for coupon distribution
    with col4:
      st.write('**Coupon Distribution**')
      chart_data = data.coupon.value_counts()
      st.bar_chart(chart_data)
    
    with col5:
    # gender wise accepted coupons
      st.write('**Gender wise accepted coupons**')
      ax = pd.crosstab(data.gender,data.Y).plot(kind='bar')
      for i in ax.containers:
        ax.bar_label(i)
      st.pyplot(ax.figure)

    with column6:
      # age wise accepted coupons
      st.write('**Age wise accepted coupons**')
      ax = pd.crosstab(data.age,data.Y).plot(kind='bar')
      for i in ax.containers:
        ax.bar_label(i)
      st.pyplot(ax.figure)
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

    # Education wise accepted coupon
    pivoted_data = data.pivot_table(index = 'education',columns = 'Y',aggfunc='size',fill_value=0)
    fig8, ax = plt.subplots(figsize=(10,5))
    pivoted_data.plot(kind='barh',stacked=True,ax=ax)
    ax.set_xlabel("Number of Coupons")
    ax.set_ylabel("Education Category")
    ax.set_title("Education Category-wise Accepted Coupons")
    for p in ax.patches:
      width, height = p.get_width(), p.get_height()
      x,y = p.get_xy()
      ax.annotate(f'{width}', (x+width/2, y+height/2), ha = 'center')
    st.pyplot(fig8)
    
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
     destination_display = ('No Urgent Place','Home','Work')
     destination_options = list(range(len(destination_display)))
     destination = st.selectbox('Destination',destination_options,format_func=lambda x: destination_display[x])

     passanger_display = ('Alone','Friend(s)','Partner','Kid(s)')
     passanger_options = list(range(len(passanger_display)))
     passanger = st.selectbox('Passanger',passanger_options,format_func=lambda x: passanger_display[x])
     
     weather_display = ('Sunny','Snowy','Rainy')
     weather_options = list(range(len(weather_display)))
     weather = st.selectbox('Weather',weather_options,format_func=lambda x: weather_display[x])
     
     temperature_display = ('80','55','30')
     temperature_options = list(range(len(temperature_display)))
     temperature = st.selectbox('Temperature',temperature_options,format_func=lambda x: temperature_display[x])

   with col2:
     time_display = ('2PM','10AM','6PM','7AM','10PM')
     time_options = list(range(len(time_display)))
     time = st.selectbox('Time',time_options,format_func=lambda x: time_display[x])
     
     coupon_display = ('Restaurant(<20)','Coffee House','Carry out & Take away','Bar','Restaurant(20-50)')
     coupon_options = list(range(len(coupon_display)))
     coupon = st.selectbox('Coupon',coupon_options,format_func=lambda x: coupon_display[x])
     
     expiration_display = ('1d','2h')
     expiration_options = list(range(len(expiration_display)))
     expiration = st.selectbox('Expiration',expiration_options,format_func=lambda x: expiration_display[x])
     
     gender_display = ('Female','Male')
     gender_options = list(range(len(gender_display)))
     gender = st.selectbox('Gender',gender_options,format_func=lambda x: gender_display[x])
    
   with col3:
     age_display = ('21','46','26','31','41','50plus','36','below21')
     age_options = list(range(len(age_display)))
     age = st.selectbox('Age',age_options,format_func=lambda x: age_display[x])
     
     maritalStatus_display = ('Unmarried partner','Single','Married partner','Divorced','Widowed')
     maritalStatus_options = list(range(len(maritalStatus_display)))
     maritalStatus = st.selectbox('Marital Status',maritalStatus_options,format_func=lambda x: maritalStatus_display[x])
     
     has_children_display = ('1','0')
     has_children_options = list(range(len(has_children_display)))
     has_children = st.selectbox('Has Children',has_children_options,format_func=lambda x: has_children_display[x])
     
     education_display = ('Some college - no degree','Bachelors degree','Associates degree','High School Graduate','Graduate degree (Masters or Doctorate)','Some High School')
     education_options = list(range(len(education_display)))
     education = st.selectbox('Education',education_options,format_func=lambda x: education_display[x])

   with col4:
     occupation_display = ('Unemployed','Architecture & Engineering','Student','Education&Training&Library',
     'Healthcare Support','Sales & Related','Management','Arts Design Entertainment Sports & Media',
     'Computer & Mathematical','Life Physical Social Science','Personal Care & Service','Community & Social Services',
     'Office & Administrative Support','Construction & Extraction','Legal','Installation Maintenance & Repair','Business & Financial',
     'Food Preparation & Serving Related','Production Occupations','Building & Grounds Cleaning & Maintenance','Transportation & Material Moving',
     'Protective Service','Healthcare Practitioners & Technical','Farming Fishing & Forestry','Retired','Military')
     occupation_options = list(range(len(occupation_display)))
     occupation = st.selectbox('Occupation',occupation_options,format_func=lambda x: occupation_display[x])
     
     income_display = ('$25000 - $37499','$12500 - $24999','$37500 - $49999','$100000 or More','$50000 - $62499','Less than $12500','$87500 - $99999','$75000 - $87499','$62500 - $74999')
     income_options = list(range(len(income_display)))
     income = st.selectbox('Income',income_options,format_func=lambda x: income_display[x])
     
     Bar_display = ('never','less1','1~3','gt8','4~8')
     Bar_options = list(range(len(Bar_display)))
     Bar = st.selectbox('Bar',Bar_options,format_func=lambda x: Bar_display[x])
     
     CoffeeHouse_display = ('never','less1','4~8','1~3','gt8')
     CoffeeHouse_options = list(range(len(CoffeeHouse_display)))
     CoffeeHouse = st.selectbox('CoffeeHouse',CoffeeHouse_options,format_func=lambda x: CoffeeHouse_display[x])

   with col5:
     CarryAway_display = ('never','less1','1~3','4~8','gt8')
     CarryAway_options = list(range(len(CarryAway_display)))
     CarryAway = st.selectbox('CarryAway',CarryAway_options,format_func=lambda x: CarryAway_display[x])
     
     RestaurantLessThan20_display = ('4~8','1~3','less1','never','gt8')
     RestaurantLessThan20_options = list(range(len(RestaurantLessThan20_display)))
     RestaurantLessThan20 = st.selectbox('RestaurantLessThan20',RestaurantLessThan20_options,format_func=lambda x: RestaurantLessThan20_display[x])
     
     Restaurant20To50_display = ('1~3','less1','never','4~8','gt8')
     Restaurant20To50_options = list(range(len(Restaurant20To50_display)))
     Restaurant20To50 = st.selectbox('Restaurant20To50',Restaurant20To50_options,format_func=lambda x: Restaurant20To50_display[x])
     
     toCoupon_GEQ15min_display = ('1','0')
     toCoupon_GEQ15min_options = list(range(len(toCoupon_GEQ15min_display)))
     toCoupon_GEQ15min = st.selectbox('toCoupon_GEQ15min',toCoupon_GEQ15min_options,format_func=lambda x: toCoupon_GEQ15min_display[x])


   with col6:
     toCoupon_GEQ15min_display = ('1','0')
     toCoupon_GEQ15min_options = list(range(len(toCoupon_GEQ15min_display)))
     toCoupon_GEQ25min = st.selectbox('toCoupon_GEQ25min',toCoupon_GEQ15min_options,format_func=lambda x: toCoupon_GEQ15min_display[x])
     
     direction_same_display = ('0','1')
     direction_same_options = list(range(len(direction_same_display)))
     direction_same = st.selectbox('direction_same',direction_same_options,format_func=lambda x: direction_same_display[x])
     
     direction_opp_display = ('0','1')
     direction_opp_options = list(range(len(direction_opp_display)))
     direction_opp = st.selectbox('direction_opp',direction_opp_options,format_func=lambda x: direction_opp_display[x])
      
   if st.button('Predict'):
    # load the model
    import pickle
    model = pickle.load(open('lr.pkl', 'rb'))

  # Get selected values from Streamlit input fields
    input_data = [
        destination, passanger, weather, temperature, time, coupon,
        expiration, gender, age, maritalStatus, has_children,
        education, occupation, income, Bar, CoffeeHouse,
        CarryAway, RestaurantLessThan20, Restaurant20To50, toCoupon_GEQ15min, 
        toCoupon_GEQ25min, direction_same, direction_opp
      ]

    # Apply model to make prediction
    prediction = model.predict([input_data])

    # Display prediction
    if prediction == 1:
      st.markdown(f"The customer likes to <b><i>accept</i></b> the <b><i>{coupon}</i></b> coupon", unsafe_allow_html=True)
    else:
      st.markdown(f"The customer likes to <b><i>reject</i></b> the <b><i>{coupon}</i></b> coupon", unsafe_allow_html=True)
   
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