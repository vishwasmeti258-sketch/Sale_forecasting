import numpy as np
import pandas as pd
import matplotlib.pyplot as  plt
import seaborn as sb
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

#-----------------------------------------------------------------------------------------------
# Data cleaning
#-----------------------------------------------------------------------------------------------
df = pd.read_csv('csv/sales.csv')
#print(df.head())
print(df.describe())
#print(df.isnull().sum())
print(df.info())

#-----------------------------------------------------------------------------------------------
# Data manipulation
# ----------------------------------------------------------------------------------------------

# changing the datatype to datetime
df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Ship Date'] = pd.to_datetime(df['Ship Date'])
df['TT'] = df['Ship Date'] - df['Order Date']

# Average  shipment
grpby = df.groupby('Country')['TT'].mean().plot(kind='bar')
plt.show()

#print(df['Order Date'])

# getting  month.
df['Month'] = df['Order Date'].dt.month_name()
print(df['Month'])

#-----------------------------------------------------------------------------------------------
# Data visualization
#-----------------------------------------------------------------------------------------------
grpby = df.groupby('Country')['Total Revenue'].sum().nlargest(8).plot(kind='pie')
plt.title('country wise sales')
plt.show
print(grpby)

grpby = df.groupby('Region')['Total Revenue'].sum().nlargest(10 ).plot(kind='line')
Gby = df.groupby('Region')['Total Revenue'].mean().nlargest(10).plot(kind='line')
plt.title('Region wise sales')
plt.legend()
plt.show()

sb.violinplot(x='Region',y='Units Sold',data=df)
plt.title('frequency sales')
plt.show()

sb.boxplot(x='Item Type',y='Total Profit',data=df,palette='husl')
plt.title('item wise profit')
plt.show()

#medium of order

sf=df['Sales Channel'].value_counts().plot(kind='bar')
plt.title('medium of order')
plt.xlabel('medium')
plt.ylabel('counts')
plt.show()

# unit sold by per item type
grpby = df.groupby('Item Type')['Units Sold'].sum().plot(kind='bar')
plt.title('unit sold by item type')
plt.xlabel('Item type')
plt.ylabel('units sold')
plt.show()

grpby = df.groupby('Item Type')['Unit Price'].mean().plot(kind='line')
plt.title('unit price by item type')
plt.xlabel('Item type')
plt.ylabel('units sold')
plt.show()

#------------------------------------------------------------------------------------------------
#Arima model for sales predication
#------------------------------------------------------------------------------------------------
adf = adfuller(df['Total Revenue'])
print('ADF statistics::',adf[0])
print('p-value::',adf[1])

model = ARIMA(df['Total Revenue'],order=(1,1,1))
model.fit = model.fit()
print(model.fit.summary)

forecast = model.fit.forecast(steps=6)
plt.plot(df['Total Revenue'],label='HisSales')
plt.plot(forecast,label='forecast',linestyle='--')
plt.ylabel('Sales')
plt.xlabel('count')
plt.legend()
plt.show()

#--------------------------------------------------------------------------------------------------
# data Saving to csv file
df.to_csv('Nsales.csv',index=False)