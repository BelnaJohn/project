#!/usr/bin/env python
# coding: utf-8

# In[1]:


from datetime import date, datetime, timedelta
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
#from statsmodels.tsa.api import Holt,SimpleExpSmoothing,ExponentialSmoothing
from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error,r2_score
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# In[2]:


url1="https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"


# In[3]:


url2="https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"


# In[4]:


url3="https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv"


# In[5]:


cnfmd=pd.read_csv(url2)
death=pd.read_csv(url1)
recvd=pd.read_csv(url3)


# In[6]:


cnfmd.columns = list(cnfmd.columns[:4]) + [datetime.strptime(d, "%m/%d/%y").date().strftime("%Y-%m-%d") for d in cnfmd.columns[4:]]
death.columns    = list(death.columns[:4])+ [datetime.strptime(d, "%m/%d/%y").date().strftime("%Y-%m-%d") for d in death.columns[4:]]
recvd.columns    = list(recvd.columns[:4])+ [datetime.strptime(d, "%m/%d/%y").date().strftime("%Y-%m-%d") for d in recvd.columns[4:]]


# In[7]:


#removing some problematic data and renaming country or region

removed_states = "Recovered|Grand Princess|Diamond Princess"
removed_countries = "US|The West Bank and Gaza"

cnfmd.rename(columns={"Province/State": "Province_State", "Country/Region": "Country_Region"}, inplace=True)
death.rename(columns={"Province/State": "Province_State", "Country/Region": "Country_Region"}, inplace=True)
recvd.rename(columns={"Province/State": "Province_State", "Country/Region": "Country_Region"}, inplace=True)

cnfmd = cnfmd[~cnfmd["Province_State"].replace(np.nan, "nan").str.match(removed_states)]
death = death[~death["Province_State"].replace(np.nan, "nan").str.match(removed_states)]
recvd = recvd[~recvd["Province_State"].replace(np.nan, "nan").str.match(removed_states)]
cnfmd = cnfmd[~cnfmd["Country_Region"].replace(np.nan, "nan").str.match(removed_countries)]
death = death[~death["Country_Region"].replace(np.nan, "nan").str.match(removed_countries)]
recvd = recvd[~recvd["Country_Region"].replace(np.nan, "nan").str.match(removed_countries)]

cnfmd.drop(columns=["Lat", "Long"], inplace=True)
death.drop(columns=["Lat", "Long"], inplace=True)
recvd.drop(columns=["Lat","Long"],inplace=True)


# In[8]:


cnfmd.head()


# In[9]:


recvd.head()


# before going further let us see what is global trend now

# In[10]:


cnfmd_ts=cnfmd.copy().drop(['Country_Region','Province_State'],axis=1)


# In[11]:


cnfmd_ts_summary=cnfmd_ts.sum()


# In[12]:


cnfmd_ts_summary


# In[13]:


fig_1=go.Figure(data=go.Scatter(x=cnfmd_ts_summary.index,y=cnfmd_ts_summary.values))
fig_1.update_layout(title="total corona viruses confirmed globally",yaxis_title="confirmed cases",xaxis_tickangle=315)
fig_1.show()
            


# total corona virus caases is about to reach one million soon

# In[14]:


recvd_ts=recvd.copy().drop(['Country_Region','Province_State'],axis=1)


# In[15]:


recvd_ts_summary=recvd_ts.sum()


# In[16]:


recvd_ts_summary


# In[17]:


fig_2=go.Figure(data=go.Scatter(x=recvd_ts_summary.index,y=recvd_ts_summary.values))
fig_2.update_layout(title="total corona viruses recovered globally",yaxis_title="recovered cases",xaxis_tickangle=315)
fig_2.show()
            


# In[18]:


death_ts=death.copy().drop(['Country_Region','Province_State'],axis=1)
death_ts_summary=death_ts.sum()
death_ts_summary


# In[19]:


fig_3=go.Figure(data=go.Scatter(x=death_ts_summary.index,y=death_ts_summary.values))
fig_3.update_layout(title="total death cases globally",yaxis_title="death cases",xaxis_tickangle=315)
fig_3.show()


# total death cases reached 1.6 million as per jan 19 2021

# cleaning and pulling indian data from the datasets

# In[20]:


#imputing missing values
for col in cnfmd.columns[2:]:
    cnfmd[col].fillna(0, inplace=True)
    death[col].fillna(0, inplace=True)
    recvd[col].fillna(0, inplace=True)


# In[21]:


cnfmd_melted = cnfmd.melt(cnfmd.columns[:2], cnfmd.columns[2:], "Date", "ConfirmedCases")
#confirmed_melted.insert(5, "Type", "Confirmed")
death_melted = death.melt(death.columns[:2], death.columns[2:], "Date", "Deaths")
#deaths_melted.insert(5, "Type", "Deaths")


cnfmd_melted.sort_values(by=["Country_Region", "Province_State", "Date"], inplace=True)
death_melted.sort_values(by=["Country_Region", "Province_State", "Date"], inplace=True)

assert cnfmd_melted.shape==death_melted.shape
assert list(cnfmd_melted["Province_State"])==list(death_melted["Province_State"])
assert list(cnfmd_melted["Country_Region"])==list(death_melted["Country_Region"])
assert list(cnfmd_melted["Date"])==list(death_melted["Date"])

cases = cnfmd_melted.merge(death_melted, on=["Province_State", "Country_Region", "Date"], how="inner")
cases = cases[["Country_Region", "Province_State", "Date", "ConfirmedCases", "Deaths"]]

cases.sort_values(by=["Country_Region", "Province_State", "Date"], inplace=True)
cases.insert(0, "Id", range(1, cases.shape[0]+1))
cases


# In[22]:


recvd_melted = recvd.melt(recvd.columns[:2], recvd.columns[2:], "Date", "Recovered")
recvd_melted.sort_values(by=["Country_Region", "Province_State", "Date"], inplace=True)


# In[23]:


rcvd_india=recvd_melted[recvd_melted['Country_Region']=='India']
rcvd_india


# In[24]:


covid_india=cases[cases['Country_Region']=='India']
covid_india


# In[25]:


assert list(covid_india["Province_State"])==list(rcvd_india["Province_State"])
assert list(covid_india["Country_Region"])==list(rcvd_india["Country_Region"])
assert list(covid_india["Date"])==list(covid_india["Date"])


data = covid_india.merge(rcvd_india,on=["Province_State", "Country_Region", "Date"], how="inner")
data = data[["Country_Region", "Province_State", "Date", "ConfirmedCases", "Deaths","Recovered"]]


data


# In[26]:


india_datewise=data.groupby(["Date"]).agg({"ConfirmedCases":'sum',"Recovered":'sum',"Deaths":'sum'})

india_datewise.head()


# In[27]:


india_datewise["Days_since"]=np.arange(len(india_datewise))


# In[28]:


india_datewise


# current scenario in India

# In[29]:


print("Number of Confirmed Cases",india_datewise["ConfirmedCases"].iloc[-1])
print("Number of Recovered Cases",india_datewise["Recovered"].iloc[-1])
print("Number of Death Cases",india_datewise["Deaths"].iloc[-1])
print("Number of Active Cases",india_datewise["ConfirmedCases"].iloc[-1]-india_datewise["Recovered"].iloc[-1]-india_datewise["Deaths"].iloc[-1])
print("Number of Closed Cases",india_datewise["Recovered"].iloc[-1]+india_datewise["Deaths"].iloc[-1])
print("Approximate Number of Confirmed Cases per day",round(india_datewise["ConfirmedCases"].iloc[-1]/india_datewise.shape[0]))
print("Approximate Number of Recovered Cases per day",round(india_datewise["Recovered"].iloc[-1]/india_datewise.shape[0]))
print("Approximate Number of Death Cases per day",round(india_datewise["Deaths"].iloc[-1]/india_datewise.shape[0]))
print("Number of New Cofirmed Cases in last 24 hours are",india_datewise["ConfirmedCases"].iloc[-1]-india_datewise["ConfirmedCases"].iloc[-2])
print("Number of New Recoverd Cases in last 24 hours are",india_datewise["Recovered"].iloc[-1]-india_datewise["Recovered"].iloc[-2])
print("Number of New Death Cases in last 24 hours are",india_datewise["Deaths"].iloc[-1]-india_datewise["Deaths"].iloc[-2])


# In[30]:


fig=px.bar(x=india_datewise.index,y=india_datewise["ConfirmedCases"]-india_datewise["Recovered"]-india_datewise["Deaths"])
fig.update_layout(title="Distribution of Number of Active Cases",xaxis_title="Date",yaxis_title="Number of Cases")
fig.show()


# In[31]:


fig=px.bar(x=india_datewise.index,y=india_datewise["Recovered"]+india_datewise["Deaths"])
fig.update_layout(title="Distribution of Number of Closed Cases",
                  xaxis_title="Date",yaxis_title="Number of Cases")
fig.show()


# In[32]:


fig=px.bar(x=india_datewise.index,y=india_datewise["Deaths"])
fig.update_layout(title="Distribution of Number of Death Cases",
                  xaxis_title="Date",yaxis_title="Number of Cases")
fig.show()


# death cases is still increasing,but the graph showing a tendency to become flat and then decrease soon
# 

# In[33]:


fig=go.Figure()
fig.add_trace(go.Scatter(x=india_datewise.index, y=india_datewise["ConfirmedCases"],
                    mode='lines+markers',
                    name='Confirmed Cases'))
fig.add_trace(go.Scatter(x=india_datewise.index, y=india_datewise["Recovered"],
                    mode='lines+markers',
                    name='Recovered Cases'))
fig.add_trace(go.Scatter(x=india_datewise.index, y=india_datewise["Deaths"],
                    mode='lines+markers',
                    name='Death Cases'))
fig.update_layout(title="Growth of different types of cases in India",
                 xaxis_title="Date",yaxis_title="Number of Cases",legend=dict(x=0,y=1,traceorder="normal"))
fig.show()


# In[34]:


fig=go.Figure()
fig.add_trace(go.Scatter(x=india_datewise.index, y=india_datewise["ConfirmedCases"].diff().fillna(0),
                    mode='lines+markers',
                    name='Confirmed Cases'))
fig.add_trace(go.Scatter(x=india_datewise.index, y=india_datewise["Recovered"].diff().fillna(0),
                    mode='lines+markers',
                    name='Recovered Cases'))
fig.add_trace(go.Scatter(x=india_datewise.index, y=india_datewise["Deaths"].diff().fillna(0),
                    mode='lines+markers',
                    name='Death Cases'))
fig.update_layout(title="Daily increase in different types of cases in India",
                 xaxis_title="Date",yaxis_title="Number of Cases",legend=dict(x=0,y=1,traceorder="normal"))
fig.show()


# daily trend shows confirmed cases is showing a decreasing trend after reaching its peak

# now, let us move to prediction

# polynomial regression

# In[35]:


#splittig into train and test
train_ml=india_datewise.iloc[:int(india_datewise.shape[0]*0.95)]
test_ml=india_datewise.iloc[int(india_datewise.shape[0]*0.95):]


# In[36]:


poly = PolynomialFeatures(degree = 6) 


# In[37]:


train_poly=poly.fit_transform(np.array(train_ml["Days_since"]).reshape(-1,1))
test_poly=poly.fit_transform(np.array(test_ml["Days_since"]).reshape(-1,1))
y=train_ml["Deaths"]


# In[38]:


linreg=LinearRegression(normalize=True)
linreg.fit(train_poly,y)


# In[39]:


prediction_poly=linreg.predict(test_poly)
rmse_poly=np.sqrt(mean_squared_error(test_ml["Deaths"],prediction_poly))
model_scores=[]
model_scores.append(rmse_poly)
print("Root Mean Squared Error for Polynomial Regression: ",rmse_poly)   
#r2 = r2_score(test_ml['Deaths'], prediction_poly)
#print(r2)


# In[40]:


prediction_poly


# In[41]:


a=poly.fit_transform(np.array(train_ml["Days_since"]).reshape(-1,1))
b=linreg.predict(a)


# In[42]:



fig=go.Figure()
fig.add_trace(go.Scatter(x=india_datewise.index, y=india_datewise["Deaths"],
                    mode='lines+markers',name="Train Data for Death Cases"))
fig.add_trace(go.Scatter(x=india_datewise.index, y=b,
                    mode='lines',name="Polynomial Regression Best Fit",
                    line=dict(color='black', dash='dot')))
fig.update_layout(title="Death Cases Polynomial Regression Prediction",
                 xaxis_title="Date",yaxis_title="Death Cases",
                 legend=dict(x=0,y=1,traceorder="normal"))
fig.show()


# 
# 
# 

# SVM

# In[43]:


train=india_datewise.iloc[:int(india_datewise.shape[0]*0.95)]
test=india_datewise.iloc[int(india_datewise.shape[0]*0.95):]


# In[44]:


#train


# In[45]:


#test


# In[46]:


svm=SVR(C=0.01,degree=7,kernel='poly')


# In[47]:


svm.fit(np.array(train["Days_since"]).reshape(-1,1),train["Deaths"])


# In[48]:


prediction_svm=svm.predict(np.array(test_ml["Days_since"]).reshape(-1,1))


# In[49]:


plt.figure(figsize=(11,6))
predictions=svm.predict(np.array(india_datewise["Days_since"]).reshape(-1,1))
fig=go.Figure()
fig.add_trace(go.Scatter(x=india_datewise.index, y=india_datewise["Deaths"],
                    mode='lines+markers',name="Train Data for mortality"))
fig.add_trace(go.Scatter(x=india_datewise.index, y=predictions,
                    mode='lines',name="Support Vector Machine Best fit Kernel",
                    line=dict(color='black', dash='dot')))
fig.update_layout(title="Death cases Support Vectore Machine Regressor Prediction",
                 xaxis_title="Date",yaxis_title="Deaths",legend=dict(x=0,y=1,traceorder="normal"))
fig.show()


# In[50]:


rmse_svm=np.sqrt(mean_squared_error(test_ml["Deaths"],prediction_svm))
model_scores.append(rmse_svm)
print("Root Mean Squared Error for Polynomial Regression: ",rmse_svm)   


# In[51]:


#test_ml


# In[52]:


#train_ml


# poisson Auto regressive prediction

# In[53]:


train_ar=india_datewise.iloc[:int(india_datewise.shape[0]*0.95)]
test_ar=india_datewise.iloc[int(india_datewise.shape[0]*0.95):]


# In[54]:


model = AR(train_ar["Deaths"])
model_fit= model.fit()


# In[55]:


print('The lag value chose is: %s' % model_fit.k_ar)
print('The coefficients of the model are:\n %s' % model_fit.params)


# In[56]:


predictions = model_fit.predict(start=len(train_ar), end=len(train_ar) + len(test_ar)-1, dynamic=False)
predictions


# In[57]:


rmse_ar=np.sqrt(mean_squared_error(test_ml["Deaths"],predictions))
model_scores.append(rmse_ar)
print("Root Mean Squared Error for Polynomial Regression: ",rmse_ar)   
#print("Root Mean Square Error for R Model: ",np.sqrt(mean_squared_error(test_ml["Deaths"],predictions)))


# In[58]:


fig=go.Figure()
fig.add_trace(go.Scatter(x=train_ml.index, y=train_ml["Deaths"],mode='lines+markers',name="Train Data for Death Cases"))
fig.add_trace(go.Scatter(x=test_ml.index, y=test_ml["Deaths"], mode='lines+markers',name="Validation Data for Death Cases",))
fig.add_trace(go.Scatter(x=test_ml.index, y=predictions,mode='lines+markers',name="Prediction of Death Cases",))
fig.update_layout(title="Death cass AR Model Prediction",xaxis_title="Date",yaxis_title="Death Cases",legend=dict(x=0,y=1,traceorder="normal"))
fig.show()


# In[59]:


model_scores


# In[60]:


#new_date=[]
new_prediction_poly=[]
for i in range(1,5):
    #new_date.append(india_datewise.index[-1]+timedelta(days=i))
    new_date_poly=poly.fit_transform(np.array(india_datewise["Days_since"].max()+i).reshape(-1,1))
    new_prediction_poly.append(linreg.predict(new_date_poly)[0])


# In[61]:


new_prediction_poly


# In[62]:


#new_date=[]
new_prediction_svm=[]
for i in range(1,5):
    #new_date.append(india_datewise.index[-1]+timedelta(days=i))
    new_prediction_svm.append(svm.predict(np.array(india_datewise["Days_since"].max()+i).reshape(-1,1))[0])


# In[63]:


new_prediction_svm


# In[64]:


AR_model_new_prediction=[]
for i in range(1,5):
    AR_model_new_prediction.append(model_fit.predict(len(test_ar)+i)[-1])
#model_predictions["AR Model Prediction"]=AR_model_new_prediction
#model_predictions.head()


# In[65]:


AR_model_new_prediction


# In[ ]:




