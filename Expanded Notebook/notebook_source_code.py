#!/usr/bin/env python
# coding: utf-8

# <h1><center> Predicting Ice Cream Revenue for Frosty Forecasts Ice Cream Co.</center></h1>

# <p align="justify">This project aims to enhance the machine learning tool for Frosty Forecasts Ice Cream Co., expanding its capabilities to predict daily revenue not only based on outside temperature but also incorporating seasonal trends. By utilizing a comprehensive dataset that includes historical data on temperature, revenue, and additional seasonal indicators such as month, day of the week, and holidays, the enhanced model will employ linear regression and gradient descent to model these complex relationships. The primary goal is to refine Frosty Forecasts Ice Cream Co.'s decision-making process regarding the deployment of ice cream trucks, optimizing operations to target the most profitable days more accurately. This refined predictive capability aims to boost operational efficiency, reduce costs, and increase revenue more effectively, thereby supporting the company’s broader success in a dynamic market environment.</p>

# <h2><center>Exploratory Data Analysis</center></h2>

# In[64]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

df = pd.read_csv('./data/ice_cream_truck_revenue_with_seasonality.csv')
df.describe()
df['date'] = pd.to_datetime(df['date'])


# <h3>Histogram Analysis of Temperature and Revenue Data</h3>

# In[27]:


# Plot histogram for Temperature
plt.figure(figsize=(10, 5))
plt.hist(df['temperature'], bins=30, color='blue', alpha=0.7)
plt.title('Histogram of Temperature')
plt.xlabel('Temperature (°F)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Plot histogram for Revenue
plt.figure(figsize=(10, 5))
plt.hist(df['revenue'], bins=30, color='green', alpha=0.7)
plt.title('Histogram of Revenue')
plt.xlabel('Revenue ($)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


# <p align="justify">The histograms above provide a visual representation of the distribution of temperatures and revenues for the ice cream truck business, which is crucial for understanding the factors influencing sales. The temperature histogram helps identify common weather conditions, potentially indicating days with higher sales volumes, while the revenue histogram reveals the sales figures' spread over the observed period. Analyzing these distributions assists in forecasting sales and strategizing business operations, highlighting the variability and central tendencies that are vital for predictive modeling and maximizing profitability.</p>

# <h3>Line Graph Analysis of Temperature and Revenue Trends Over Time</h3>

# In[36]:


# Plot line graph for Temperature vs. Date
plt.figure(figsize=(14, 7))
plt.plot(df['date'], df['temperature'], color='blue', label='Temperature')
plt.title('Temperature Trend Over Time')
plt.xlabel('Date')
plt.ylabel('Temperature (°F)')
plt.grid(True)
plt.legend()
plt.show()

# Plot line graph for Revenue vs. Date
plt.figure(figsize=(14, 7))
plt.plot(df['date'], df['revenue'], color='green', label='Revenue')
plt.title('Revenue Trend Over Time')
plt.xlabel('Date')
plt.ylabel('Revenue ($)')
plt.grid(True)
plt.legend()
plt.show()


# <p align="justify">The line graphs above provide a detailed view of how temperature and revenue fluctuate throughout the observation period. The temperature trend line helps identify periods with significant warmth that likely correlate with increased ice cream sales. Observing these patterns allows us to predict potential spikes in demand based on seasonal temperature changes. Conversely, the revenue trend line directly illustrates the sales performance over time, highlighting peak periods, possible sales slumps, and any unexpected fluctuations. Together, these visualizations are crucial for assessing the temporal dynamics of the business, offering insights into how external conditions like weather and seasonality affect sales. This analysis supports strategic planning and forecasting by pinpointing optimal times for resource allocation and promotional efforts, ultimately aiming to enhance profitability and operational efficiency.</p>

# <h3>Heatmap Analysis of Feature Correlations</h3>

# In[47]:


# Calculate the correlation matrix
corr = df.corr()

# Set up the matplotlib figure
plt.figure(figsize=(12, 8))

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=.5, cbar_kws={"shrink": .5})

# Add title
plt.title('Correlation Matrix of Features')

# Show the plot
plt.show()


# <p align="justify">The heatmap displayed above provides a visual representation of the correlation coefficients between all pairs of features within our dataset. Each cell in the heatmap shows the correlation value between two features, where the color intensity reflects the strength and the sign (positive or negative) of the correlation. Positive correlations are shown in warmer tones (red), indicating that as one feature increases, the other also tends to increase. Conversely, cooler tones (blue) denote negative correlations, suggesting that as one feature increases, the other decreases.
# 
# This analysis is crucial for identifying relationships that can influence model performance and data interpretation. For instance, a high positive correlation between temperature and revenue supports our hypothesis that warmer days lead to higher ice cream sales. On the other hand, identifying highly correlated independent features is important for avoiding multicollinearity in regression models, which can distort the estimated coefficients. By understanding these relationships, we can make informed decisions about feature selection and engineering to improve model accuracy and reliability.</p>

# In[ ]:





# In[ ]:




