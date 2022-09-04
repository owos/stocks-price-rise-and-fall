import streamlit as st
import datetime
import numpy as np
import plotly.graph_objects as go
from stock import Stock
import snscrape.modules.twitter as sntwitter #used to scrape tweets
import re
from datetime import date, timedelta # Date Functions
from tqdm import tqdm
import string
import yfinance as yf

import numpy as np # Fundamental package for scientific computing with Python
import pandas as pd # For analysing and manipulating data
from datetime import date, timedelta # Date Functions
from pandas.plotting import register_matplotlib_converters # Adds plotting functions for calender dates
import matplotlib.pyplot as plt # For visualization
import matplotlib.dates as mdates # Formatting dates
import joblib 
import keras
import snscrape.modules.twitter as sntwitter #used to scrape tweets


import flair
sentiment_model = flair.models.TextClassifier.load('en-sentiment')

st.set_page_config(layout="wide", initial_sidebar_state="expanded")
st.title('Stock forecast dashboard')



      
# ------ layout setting---------------------------
window_selection_c = st.sidebar.container() # create an empty container in the sidebar
window_selection_c.markdown("## Insights") # add a title to the sidebar container
sub_columns = window_selection_c.columns(2) #Split the container into two columns for start and end date

# ----------Time window selection-----------------
YESTERDAY=datetime.date.today()-datetime.timedelta(days=1)
YESTERDAY = Stock.nearest_business_day(YESTERDAY) #Round to business day

DEFAULT_START=YESTERDAY - datetime.timedelta(days=700)
DEFAULT_START = Stock.nearest_business_day(DEFAULT_START)

START = sub_columns[0].date_input("From", value=datetime.date(2015, 1, 1,), max_value=YESTERDAY - datetime.timedelta(days=1))
END = sub_columns[1].date_input("To", value=YESTERDAY, max_value=YESTERDAY, min_value=START)

START = Stock.nearest_business_day(START)
END = Stock.nearest_business_day(END)
# ---------------stock selection------------------
STOCKS = np.array([ 'GOOGLE', 'TESLA', 'MICROSOFT', 'META', 'COCA COLA', 'VISA', 'PFIZER', 'CHEVRON', 'WALMART', 'UNITED PARCEL SERVICES'])  # TODO : include all stocks
SYMB = window_selection_c.selectbox("select stock", STOCKS)
stocks_name = {'GOOGLE':'GOOG', 'TESLA':'TSLA', 'MICROSOFT':'MSFT', 'META':'META', 'COKA COLA':'KO', 'VISA':'V', 'PFIZER':'PFE', 
                'CHEVRON':'CVX', 'WALMART':'WMT', 'UNITED PARCEL SERVICES':'UPS'}




chart_width= st.expander(label="chart width").slider("", 500, 2800, 1000, key='CHART_WIDTH')


# # # ------------------------Plot stock linecharts--------------------

today = date.today()

date_today = today.strftime("%Y-%m-%d")
date_start = str(START)#'2015-01-01'

stock = yf.download(stocks_name[SYMB], start=date_start, end=str(END)).reset_index()

st.subheader('Prices from the last 10 days')
st.write(stock.tail(10))

## PLot for raw data
register_matplotlib_converters()
years = mdates.YearLocator()
fig, ax1 = plt.subplots(figsize=(16, 6))
ax1.xaxis.set_major_locator(years)
x = stock.index
y = stock['Close']
ax1.fill_between(x, 0, y, color='#b9e1fa')
ax1.legend([SYMB], fontsize=12)
plt.title(SYMB + ' from '+ date_start + ' to ' + date_today, fontsize=16)
plt.plot(y, color='#039dfc', label=SYMB, linewidth=1.0)
plt.ylabel('Stocks', fontsize=12)
st.pyplot(fig)

change_c = st.sidebar.container()




#----part-1--------------------------------Session state intializations---------------------------------------------------------------

if "TEST_INTERVAL_LENGTH" not in st.session_state:
    # set the initial default value of test interval
    st.session_state.TEST_INTERVAL_LENGTH = 60

if "TRAIN_INTERVAL_LENGTH" not in st.session_state:
    # set the initial default value of the training length widget
    st.session_state.TRAIN_INTERVAL_LENGTH = 500

if "HORIZON" not in st.session_state:
    # set the initial default value of horizon length widget
    st.session_state.HORIZON = 60

if 'TRAINED' not in st.session_state:
    st.session_state.TRAINED=False

#---------------------------------------------------------Train_test_forecast_splits---------------------------------------------------
st.sidebar.markdown("# Forecasts")
train_test_forecast_c = st.sidebar.container()



pred_type = window_selection_c.selectbox("Prediction Type", ['Predict Price', 'Predict Sentiment'])
@st.cache
def load_model():
	  return keras.models.load_model('stock_50epoch.h5')

if pred_type== 'Predict Price':
    train_test_forecast_c.markdown("## Make prediction")
    run = train_test_forecast_c.button(
    label="Predict",
    key='TRAIN_JOB'
    )
    if run==True:
        sequence_length = 50
        df_temp = stock[-sequence_length:]
        new_df = df_temp.filter(['Open', 'High', 'Low', 'Adj Close', 'Close'])

        N = sequence_length

        # then just 'dump' your file
        mmscaler = joblib.load('mmscaler.pkl') 
        last_N_days = new_df[-sequence_length:].values
        last_N_days_scaled = mmscaler.transform(last_N_days)

        # Create an empty list and Append past N days
        X_test_new = []
        X_test_new.append(last_N_days_scaled)
        model =  load_model()

        # Convert the X_test data set to a numpy array and reshape the data
        pred_price_scaled = model.predict(np.array(X_test_new))
        pred_price_scaled = np.repeat(pred_price_scaled.reshape(-1, 1), 5, axis=-1)
        pred_price_unscaled = mmscaler.inverse_transform(pred_price_scaled)

        # Print last price and predicted price for the next day
        price_today = np.round(new_df['Close'].values[-1], 2)
        predicted_price = np.round(pred_price_unscaled.ravel()[0], 2)
        change_percent = np.round(100 - (price_today * 100)/predicted_price, 2)

        plus = '+'; minus = ''
        st.write(f'The close price for {stocks_name[STOCKS]} at {today} was {price_today}')
        st.write(f'The predicted close price is {predicted_price} ({plus if change_percent > 0 else minus}{change_percent}%)')
        
        st.markdown("### Trend prediction")

        trend_model = joblib.load('trend_model.pkl')
        trend = trend_model.predict(new_df.tail(1))[0]
        if trend==1:
            st.write("The trend is a rise")
        else:
            st.write("The trend is a Fall")
    
elif pred_type== 'Predict Sentiment':
    query = "(from:$GOOG) until:{} since:{}".format(date_today, (today - datetime.timedelta(days = 30)))
    tweets = []
    limit = 5000

    run = train_test_forecast_c.button(
    label="Predict"
    )
    if run==True:
        for tweet in sntwitter.TwitterSearchScraper(query).get_items():
            
            # print(vars(tweet))
            # break
            if len(tweets) == limit:
                break
            else:
                tweets.append([tweet.date, tweet.username, tweet.content])
                
        df = pd.DataFrame(tweets, columns=['Date', 'User', 'Tweet'])
        stopwords= set(['br', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \
            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\
            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\
            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \
            'won', "won't", 'wouldn', "wouldn't", "u" ])
        def decontracted(phrase):
            # specific
            phrase = re.sub(r"won't", "will not", phrase)
            phrase = re.sub(r"can\'t", "can not", phrase)

            # general
            phrase = re.sub(r"n\'t", " not", phrase)
            phrase = re.sub(r"\'re", " are", phrase)
            phrase = re.sub(r"\'s", " is", phrase)
            phrase = re.sub(r"\'d", " would", phrase)
            phrase = re.sub(r"\'ll", " will", phrase)
            phrase = re.sub(r"\'t", " not", phrase)
            phrase = re.sub(r"\'ve", " have", phrase)
            phrase = re.sub(r"\'m", " am", phrase)
            phrase = re.sub(r"\'u", " you", phrase)
            return phrase


        def process_text(text):
            preprocessed_text = []
            for text in tqdm(text):
                text = re.sub(r"http\S+", "", text)
                text = re.sub('\[.*?\]', '', text)
                text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
                text = decontracted(text)
                text = re.sub('\w*\d\w*', '', text)
                text = re.sub('[‘’“”…]', '', text)
                text = re.sub('\n', '', text)
                text = re.sub("\S*\d\S*", "", text).strip()
                text = re.sub('[^A-Za-z]+', ' ', text)
                # https://gist.github.com/sebleier/554280
                text = ' '.join(e.lower() for e in text.split() if e.lower() not in stopwords)
                preprocessed_text.append(text.strip())
            return preprocessed_text 
        
        # we will append probability and sentiment preds later
        probs = []
        sentiments = []
        
        # use regex expressions (in clean function) to clean tweets
        df['clean'] = process_text(df['Tweet'])
        
        for tweet in df['clean'].to_list():
            # make prediction
            sentence = flair.data.Sentence(tweet)
            sentiment_model.predict(sentence)
            # extract sentiment prediction
            probs.append(sentence.labels[0].score)  # numerical score 0-1
            sentiments.append(sentence.labels[0].value)  # 'POSITIVE' or 'NEGATIVE'

        # add probability and sentiment predictions to tweets dataframe
        df['probability'] = probs
        df['sentiment'] = sentiments
        st.write(df)











