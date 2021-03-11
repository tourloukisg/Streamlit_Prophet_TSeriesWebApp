# STREAMLIT --- PYTHON --- MACHINE LEARNING --- Time Series Forecasting


# In this example, the use of the FB Prophet model for time series forecasting
# with respect to four stocks(Netflix, Amazon, Google & Microsoft ) is
# demonstrated. In addition, there is use of the Streamlit open-source Python 
# library to create a web app where a)stock selection option is provided, 
# b)prediction horizon selection option is also provided, c)the stock open,high,
# low, close, adj close & volume prices are included, d) basic descriptive
# statistics are presented, e) a time series plot of the selected stock is
# displayed, together with its histogram and KDE distribution plot,
# f)the last 5 forecasted Adj Close values are displayed and g)the Adj Close 
# Time Series Forecast plot & plots of the forecast components are provided

# To download the historical market data with python, there is use of the 
# Yahoo! finance market data downloader.

# NOTE:
# I)    
#      Streamlit requires a filename with the suffix .py appended. If there is
#      use of a notebook(ipynb file), then it should be saved as 'filename.py'

# II)
#      To run the Web App, a command similar to the one presented below can
#       be used (Command  Prompt, Anaconda Prompt..)
#       streamlit run "C:\Users\username\Desktop\web_app\stock_forecast.py"




# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from datetime import date
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
import yfinance as yfin
import warnings
warnings.filterwarnings('ignore')

# Creating the Web App Title
st.title('Stock Time Series Forecasting - Web App')

# User option to select the name of the stock for time series forecasting 
available_stocks=('NFLX','AMZN','GOOGL','MSFT')
select_stock=st.selectbox('Select Stock for Time Series Forecasting',
                          available_stocks)

# Selection of the prediction horizon in years (1 to 5 years ahead)
pred_hor=st.slider("Prediction Horizon Selection (Years Ahead):",1,5)

# Selecting the dataset start and end date 
start_date='2011-01-01'
end_date=date.today().strftime("%Y-%m-%d")

# Function to download the selected stock dataset and save it to cache 
#(avoiding stock re-download) 
@st.cache
def get_stock_data(stock):
    dataset=yfin.download(stock,start_date,end_date)
    dataset.reset_index(inplace=True)
    return dataset

# Display message while downloading the selected stock dataset 
stock_dload=st.text('Downloading the selected Stock Dataset. . . . . .')
stock_data=get_stock_data(select_stock)
stock_dload.text('The Dataset has been downloaded !!!')

# Observing the last 5 trading days
st.subheader('{} Dataset - Last 5 Trading Days'.format(select_stock))
st.write(stock_data.tail())
# Observing basic statistical details of the stock
st.subheader('{} Dataset - Descriptive Statistics'.format(select_stock))
st.write(stock_data.describe().transpose())

# Plotting the Ajd.Close time series of the selected stock
st.subheader('{} Dataset - Time Series Plot'.format(select_stock))
def timeseriesplot():
    fig=plt.figure(figsize=(10,6))
    plt.plot(stock_data['Date'],stock_data['Adj Close'],c='magenta',
             linestyle='dashed',label='Trading Days')
    plt.xlabel('Date',fontweight='bold',fontsize=12)
    plt.ylabel('Adj Close',fontweight='bold',fontsize=12)
    st.pyplot(fig)
timeseriesplot()
# Selected stock: Ajd.Close Histogram and KDE plot
st.subheader('{} Dataset - Stock Adj Close Histogram & KDE Plot'.
             format(select_stock))
def stockplots():
    fig,axs=plt.subplots(1,2,figsize=(14,6))
    stock_data['Adj Close'].plot(label='Adj Close Hist.',
                                 kind='hist',bins=10,ax=axs[0])
    axs[0].set_ylabel('Frequency',fontweight='bold')
    stock_data['Adj Close'].plot(label='Adj Close Dist.',c='darkorange',
                                 kind='kde',ax=axs[1])
    axs[1].set_ylabel('Density',fontweight='bold')
    for ax in axs.flat:
        plt.rcParams['font.size']=14
        ax.set_xlabel('Adj Close',fontweight='bold')
        ax.legend()
        ax.figure.tight_layout(pad=2)
    st.pyplot(fig)
stockplots()


# New dataframe that contains only the Date and Adj Close columns 
# of the selected stock as this is the required 'Prophet' model format
# with respect to model training
df_prophet_train=stock_data[['Date','Adj Close']]
df_prophet_train=df_prophet_train.rename(columns={"Date":"ds","Adj Close":"y"})

# Creating and fiting the 'Prophet' model to the training set
# Interval Width -> Uncertainty interval width (95%)
# changepoint_prior_scale -> The higher its value, the higher the value of
# forecast uncertainty.
model=Prophet(interval_width=0.95,changepoint_prior_scale=1)
model.fit(df_prophet_train)

# Adding the future Dates (1 year ~ 252 trading days)
# Setting the frequency to Business day ('B') to avoid predictions on weekends
future_dates=model.make_future_dataframe(periods=252*pred_hor,freq='B')
# Forecasting the Adj Close price for the selected time period
prophet_forecast=model.predict(future_dates)

# Observing the last 5 forecasted Adj Close values  
st.subheader("{} Dataset - Last 5 'Prophet' model Predictions".format(
    select_stock))
st.write(prophet_forecast.tail())

# Adj Close Time Series Forecast plot & plots of the forecast components
st.subheader("{} Dataset - Forecast plot (Years Ahead: {})".format(select_stock,
                                                                pred_hor))
fig_forecast = plot_plotly(model, prophet_forecast)
st.plotly_chart(fig_forecast)
st.subheader("{} Dataset - Time Series Forecast Components".format(
    select_stock))
fig_comp = model.plot_components(prophet_forecast)
st.write(fig_comp)
