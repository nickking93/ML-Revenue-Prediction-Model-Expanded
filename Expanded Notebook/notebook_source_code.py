#!/usr/bin/env python
# coding: utf-8

# <h1><center> Predicting Ice Cream Revenue for Frosty Forecasts Ice Cream Co.</center></h1>

# <p align="justify">This project aims to enhance the machine learning tool for Frosty Forecasts Ice Cream Co., expanding its capabilities to predict daily revenue not only based on outside temperature but also incorporating seasonal trends. By utilizing a comprehensive dataset that includes historical data on temperature, revenue, and additional seasonal indicators such as month, day of the week, and holidays, the enhanced model will employ linear regression and gradient descent to model these complex relationships. The primary goal is to refine Frosty Forecasts Ice Cream Co.'s decision-making process regarding the deployment of ice cream trucks, optimizing operations to target the most profitable days more accurately. This refined predictive capability aims to boost operational efficiency, reduce costs, and increase revenue more effectively, thereby supporting the company’s broader success in a dynamic market environment.</p>

# <h2><center>Dataset Description and Creation Process</center></h2>
# <p align="justify">This dataset includes 500 entries, each representing a day's data. Key features in the dataset include temperature, date, and several time-related attributes such as month, day of the week, day of the year, and week of the year. To simulate realistic conditions in Independence, MO, daily temperature values were generated around monthly average temperatures for the region, with variations of ±10 degrees. Revenue values were then generated to be realistically correlated with temperature, reflecting the natural relationship between warmer days and higher ice cream sales. Additional features such as rolling temperature averages and temperature differences from the previous day were also included to capture trends and changes over time. This comprehensive dataset provides a robust foundation for building and evaluating the predictive model.</p>

# <h2><center>Exploratory Data Analysis</center></h2>

# In[1365]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta
import requests
import random
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

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
df['date'] = pd.to_datetime(df['date'])
X = df.drop('revenue', axis=1)
y = df['revenue']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

df.describe()


# <h3>Histogram Analysis of Temperature and Revenue Data</h3>

# In[1297]:


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

# In[1300]:


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

# In[1303]:


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

# <h2><center>Linear Regression</center></h2>

# In[1306]:


# Initialize the Linear Regression model
model = LinearRegression()

# Fit the model on the training data
model.fit(X_train.drop('date', axis=1), y_train)  # Ensure to drop non-numeric columns like 'date'

# Make predictions on both training and testing sets
train_preds = model.predict(X_train.drop('date', axis=1))
test_preds = model.predict(X_test.drop('date', axis=1))


# <h3>Scatter Plot Analysis: Actual vs. Predicted Revenue</h3>

# In[1308]:


# Create a scatter plot of the actual values
plt.figure(figsize=(14, 7))
plt.scatter(X_test['date'], y_test, color='red', label='Actual Revenue', alpha=0.6)

# Create a line plot for the predicted values
plt.plot(X_test['date'], test_preds, color='blue', linewidth=2, label='Predicted Revenue')

# Formatting the plot
plt.title('Comparison of Actual and Predicted Revenue Over Time')
plt.xlabel('Date')
plt.ylabel('Revenue ($)')
plt.legend()
plt.grid(True)

# Improve readability of the x-axis
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gcf().autofmt_xdate()  # Rotation

plt.show()


# <p align="justify">This scatter plot contrasts actual revenues against predicted revenues over time, highlighting the model’s effectiveness and pinpointing outliers. The prediction line, meant to mirror the actual revenue trajectory, serves as a benchmark for assessing model accuracy; closeness of this line to the actual data points indicates robust performance. Significant deviations, or outliers, suggest areas where the model may falter, potentially due to unaccounted variables or incomplete feature representation. Analyzing these discrepancies helps refine the predictive capabilities of the model, ensuring better alignment with real-world data and enhancing decision-making processes for strategic business operations.</p>

# In[1310]:


residuals = y_test - test_preds
plt.figure(figsize=(10, 5))
plt.scatter(X_test['date'], residuals)
plt.title('Residual Plot')
plt.xlabel('Date')
plt.ylabel('Residuals')
plt.axhline(y=0, color='red', linestyle='--')
plt.grid(True)
plt.show()


# <p align="justify">
#     The scatterplot above helps to assess the quality of the regression model by checking for patterns in the residuals. The residuals are randomly scattered around the horizontal line, indicating that the model's errors are randomly distributed and that there are no systematic patterns left unexplained by the model.
# </p>

# In[1312]:


# Evaluate the model
train_rmse = root_mean_squared_error(y_train, train_preds)
test_rmse = root_mean_squared_error(y_test, test_preds)
train_r2 = r2_score(y_train, train_preds)
test_r2 = r2_score(y_test, test_preds)
scores = cross_val_score(model, X_train.drop('date', axis=1), y_train, cv=5, scoring='neg_mean_squared_error')

print(f"Training R-squared: {train_r2:.2f}")
print(f"Testing R-squared: {test_r2:.2f}")
print(f"Training RMSE: {train_rmse:.2f}")
print(f"Testing RMSE: {test_rmse:.2f}")
print("Cross-validated RMSE: ", np.sqrt(-scores.mean()))


# <p align="justify">The model's performance is evaluated using several metrics. The Training RMSE (Root Mean Squared Error) of 48.37 indicates the average error magnitude between the predicted and actual revenue on the training data, with lower values indicating better fit. The Testing RMSE of 47.77 shows the model's prediction error on unseen test data, providing an estimate of its generalization ability. The Training R-squared of 0.91 signifies that 91% of the variance in revenue is explained by the model on the training data, demonstrating a strong fit. The Testing R-squared of 0.81 indicates that the model explains 81% of the variance in the test data, confirming its effectiveness though slightly less robust than on the training set. Finally, the Cross-validated RMSE of 51.94 offers an average prediction error across multiple folds of the data, further validating the model's performance and reliability.</p>

# <h3>Parameter Tuning</h3>

# In[1315]:


# Use a grid search to find best estimator
parameters = {'fit_intercept':[True,False]}
grid_search = GridSearchCV(model, parameters, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train.drop('date', axis=1), y_train)
best_model = grid_search.best_estimator_


# In[1316]:


# Output the best parameters
print("Best parameters found: ", grid_search.best_params_)

# Output the best score
best_score = np.sqrt(-grid_search.best_score_)
print("Best cross-validated RMSE using the best parameters: {:.2f}".format(best_score))

# Apply the best model to make predictions
best_model = grid_search.best_estimator_

# Make predictions using the best model
best_train_preds = best_model.predict(X_train.drop('date', axis=1))
best_test_preds = best_model.predict(X_test.drop('date', axis=1))

# Evaluate the best model
best_train_rmse = root_mean_squared_error(y_train, best_train_preds)
best_test_rmse = root_mean_squared_error(y_test, best_test_preds)
best_train_r2 = r2_score(y_train, best_train_preds)
best_test_r2 = r2_score(y_test, best_test_preds)


# <h3>Visualization of Feature Importance in Linear Regression</h3>

# In[1318]:


# Calculate feature importance from the model coefficients
feature_importance = pd.Series(index=X_train.columns.drop('date'), data=model.coef_)

# Sort the features based on their importance
sorted_feature_importance = feature_importance.sort_values()

# Create a bar plot for feature importance
plt.figure(figsize=(12, 6))
sorted_feature_importance.plot(kind='bar', color='skyblue')
plt.title('Feature Importance in Linear Regression Model')
plt.xlabel('Features')
plt.ylabel('Coefficient Value')
plt.grid(True)
plt.show()


# <p align="justify">This bar chart illustrates the importance of each feature used in our linear regression model. Each bar represents a feature's coefficient, indicating how much the dependent variable (revenue) is expected to increase or decrease with a one-unit increase in that feature, assuming all other features hold constant.</p>
# 
# <h3>Key Aspects of the Visualization:</h3>
# 
# <ul>
#     <li>
#     Positive vs. Negative Values: Positive coefficients indicate a positive relationship with the dependent variable; as the feature increases, the revenue is expected to increase. Conversely, negative coefficients suggest that as the feature increases, the revenue is likely to decrease.
#     </li>
#     <li>
#     Magnitude of the Coefficients: The length of the bars reflects the magnitude of each feature's impact. Larger bars (whether positive or negative) signify a stronger influence on the revenue outcome. This helps in identifying which features are most influential and which might be less significant.
#     </li>
#     <li>
#     Comparative Analysis: By comparing the bars, stakeholders can quickly discern which factors are driving increases in revenue, which are detrimental, and which are neutral. This information is crucial for making informed decisions about business strategies, marketing, and operational adjustments.
#     </li>
# </ul>
# 
# <h3>Implications for Decision Making:</h3>
# 
# <p align="justify">
#     Understanding these dynamics allows business leaders at Frosty Forecasts Ice Cream Co. to refine their strategies. For instance, if temperature has a large positive coefficient, it confirms the hypothesis that warmer days significantly boost ice cream sales, underscoring the importance of weather-based marketing and staffing strategies. Alternatively, if a feature like a particular day of the week shows a negative impact, this could inform operational planning, such as reducing hours or inventory on those days to cut costs.
# 
# This feature importance plot not only informs model interpretation but also aids in the practical application of the model's findings to enhance business practices and drive profitability.<p>

# <h2><center>Random Forest</center></h2>

# In[1321]:


# Initialize the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the Random Forest model on the training data
rf_model.fit(X_train.drop('date', axis=1), y_train)

# Make predictions on both training and testing sets
train_preds_rf = rf_model.predict(X_train.drop('date', axis=1))
test_preds_rf = rf_model.predict(X_test.drop('date', axis=1))

# Evaluate the Random Forest model
train_rmse_rf = root_mean_squared_error(y_train, train_preds_rf)
test_rmse_rf = root_mean_squared_error(y_test, test_preds_rf)
train_r2_rf = r2_score(y_train, train_preds_rf)
test_r2_rf = r2_score(y_test, test_preds_rf)


print(f"Random Forest - Training R-squared: {train_r2_rf:.2f}")
print(f"Random Forest - Testing R-squared: {test_r2_rf:.2f}")
print(f"Random Forest - Training RMSE: {train_rmse_rf:.2f}")
print(f"Random Forest - Testing RMSE: {test_rmse_rf:.2f}")

# Plot the predictions for Random Forest
plt.figure(figsize=(14, 7))
plt.scatter(X_test['date'], y_test, color='red', label='Actual Revenue', alpha=0.6)
plt.plot(X_test['date'], test_preds_rf, color='blue', linewidth=2, label='Predicted Revenue (Random Forest)')

plt.title('Comparison of Actual and Predicted Revenue Over Time (Random Forest)')
plt.xlabel('Date')
plt.ylabel('Revenue ($)')
plt.legend()
plt.grid(True)
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gcf().autofmt_xdate()
plt.show()

# Create a figure with two subplots side by side
plt.figure(figsize=(20, 7))

# Subplot for Linear Regression
plt.subplot(1, 2, 1)
plt.scatter(X_test['date'], y_test, color='red', label='Actual Revenue', alpha=0.6)
plt.plot(X_test['date'], test_preds, color='blue', linewidth=2, label='Predicted Revenue (Linear)')
plt.title('Actual vs Predicted Revenue (Linear Regression)')
plt.xlabel('Date')
plt.ylabel('Revenue ($)')
plt.legend()
plt.grid(True)
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gcf().autofmt_xdate()

# Subplot for Random Forest
plt.subplot(1, 2, 2)
plt.scatter(X_test['date'], y_test, color='red', label='Actual Revenue', alpha=0.6)
plt.plot(X_test['date'], test_preds_rf, color='blue', linewidth=2, label='Predicted Revenue (Random Forest)')
plt.title('Actual vs Predicted Revenue (Random Forest)')
plt.xlabel('Date')
plt.ylabel('Revenue ($)')
plt.legend()
plt.grid(True)
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gcf().autofmt_xdate()

plt.tight_layout()
plt.show()


# <h2><center>Gradient Boosting Regression</center></h2>

# In[1323]:


# Initialize the Gradient Boosting model
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)

# Fit the Gradient Boosting model on the training data
gb_model.fit(X_train.drop('date', axis=1), y_train)

# Make predictions on both training and testing sets
train_preds_gb = gb_model.predict(X_train.drop('date', axis=1))
test_preds_gb = gb_model.predict(X_test.drop('date', axis=1))

# Evaluate the Gradient Boosting model
train_rmse_gb = root_mean_squared_error(y_train, train_preds_gb)
test_rmse_gb = root_mean_squared_error(y_test, test_preds_gb)
train_r2_gb = r2_score(y_train, train_preds_gb)
test_r2_gb = r2_score(y_test, test_preds_gb)

print(f"Gradient Boosting - Training R-squared: {train_r2_gb:.2f}")
print(f"Gradient Boosting - Testing R-squared: {test_r2_gb:.2f}")
print(f"Gradient Boosting - Training RMSE: {train_rmse_gb:.2f}")
print(f"Gradient Boosting - Testing RMSE: {test_rmse_gb:.2f}")

# Create a figure with two subplots side by side
plt.figure(figsize=(20, 7))

# Subplot for Linear Regression
plt.subplot(1, 2, 1)
plt.scatter(X_test['date'], y_test, color='red', label='Actual Revenue', alpha=0.6)
plt.plot(X_test['date'], test_preds, color='blue', linewidth=2, label='Predicted Revenue (Linear)')
plt.title('Actual vs Predicted Revenue (Linear Regression)')
plt.xlabel('Date')
plt.ylabel('Revenue ($)')
plt.legend()
plt.grid(True)
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gcf().autofmt_xdate()

# Subplot for Gradient Boosting
plt.subplot(1, 2, 2)
plt.scatter(X_test['date'], y_test, color='red', label='Actual Revenue', alpha=0.6)
plt.plot(X_test['date'], test_preds_gb, color='blue', linewidth=2, label='Predicted Revenue (Gradient Boosting)')
plt.title('Actual vs Predicted Revenue (Gradient Boosting)')
plt.xlabel('Date')
plt.ylabel('Revenue ($)')
plt.legend()
plt.grid(True)
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gcf().autofmt_xdate()

plt.tight_layout()
plt.show()


# <h3>Comparing the Models</h3>

# <p align="justify">The three models—Linear Regression, Random Forest, and Gradient Boosting—produced varying results, each with distinct performance characteristics. The Linear Regression model showed consistent performance with a Training R-squared of 0.91 and a Testing R-squared of 0.81, and RMSE values of 48.37 (training) and 47.77 (testing), indicating good generalization. The Random Forest model had a high Training R-squared of 0.98, but a lower Testing R-squared of 0.76, and RMSE values of 19.90 (training) and 53.49 (testing), suggesting overfitting. The Gradient Boosting model achieved a Training R-squared of 0.96 and a Testing R-squared of 0.79, with RMSE values of 30.64 (training) and 50.81 (testing), balancing between the two models but still indicating slight overfitting.</p>
# <p align="justify">Overall, Linear Regression provided the most stable performance across both training and testing datasets, while Random Forest and Gradient Boosting offered better training fits but struggled with generalization.</p>

# <h2><center>Visualizations to Assess Model Performance</center></h2>

# In[1327]:


# Seasonal Trend Analysis: Actual vs. Predicted Revenue for Different Seasons
df['predicted_revenue'] = best_linear_model.predict(X.drop('date', axis=1))
df['month'] = df['date'].dt.month
seasons = {
    'Winter': [12, 1, 2],
    'Spring': [3, 4, 5],
    'Summer': [6, 7, 8],
    'Fall': [9, 10, 11]
}

plt.figure(figsize=(14, 7))
for season, months in seasons.items():
    season_data = df[df['month'].isin(months)]
    plt.scatter(season_data['date'], season_data['revenue'], alpha=0.5, label=f'Actual Revenue ({season})')
    plt.scatter(season_data['date'], season_data['predicted_revenue'], alpha=0.5, label=f'Predicted Revenue ({season})', marker='x')

plt.title('Actual vs Predicted Revenue by Season')
plt.xlabel('Date')
plt.ylabel('Revenue ($)')
plt.legend()
plt.grid(True)
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gcf().autofmt_xdate()
plt.show()

# Comparison of Actual vs. Predicted Revenue Over Time
plt.figure(figsize=(14, 7))
plt.plot(df['date'], df['revenue'], color='red', label='Actual Revenue', alpha=0.6)
plt.plot(df['date'], df['predicted_revenue'], color='blue', label='Predicted Revenue', alpha=0.6)

plt.title('Comparison of Actual and Predicted Revenue Over Time')
plt.xlabel('Date')
plt.ylabel('Revenue ($)')
plt.legend()
plt.grid(True)
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gcf().autofmt_xdate()
plt.show()


# <p align="justify">The figures above provide a detailed assessment of the model's performance in predicting ice cream sales. The scatter plots display actual and predicted revenue for different seasons (Winter, Spring, Summer, Fall), with separate markers for each season's actual and predicted values. This visualization helps to evaluate how well the model captures seasonal trends. Additionally, the line plots compare actual and predicted revenue over the entire observation period, where the red line represents actual revenue and the blue line represents predicted revenue. This helps to assess the overall performance of the model in capturing revenue trends. The alignment of the predicted values with the actual values indicates the model's effectiveness, while deviations highlight areas for potential improvement.</p>

# <h2><center>Final Model Evaluation</center></h2>

# In[1330]:


#K-fold cross-validation

# Define the number of folds for k-fold cross-validation
k = 5

# Perform k-fold cross-validation
cv_scores = cross_val_score(best_linear_model, X.drop('date', axis=1), y, cv=k, scoring='neg_mean_squared_error')
cv_rmse_scores = np.sqrt(-cv_scores)

print(f"{k}-Fold Cross-Validation Scores:")
print(f"RMSE: {cv_rmse_scores}")
print(f"Mean RMSE: {cv_rmse_scores.mean()}")
print(f"Standard Deviation of RMSE: {cv_rmse_scores.std()}")


# <p align="justify">After k-fold cross-validation with 5 folds, the model yields a mean RMSE of 70.75, indicating that, on average, the model's predictions deviate from the actual values by approximately $70.75. The standard deviation of the RMSE is 39.69, highlighting the variability in the model's performance across different folds. A standard deviation of 39.69 is relatively moderate compared to the mean RMSE of 70.75. This suggests that while the model's accuracy is moderate, the variations in its predictions across different data subsets are not considered significant.</p>

# <h2><center>Conclusion</center></h2>

# <p align="justify">The initial linear regression model demonstrates strong performance with high R-squared values and low RMSE on both training and testing sets. This, along with its simplicity and interpretability, justifies the choice of linear regression over the more complex gradient boosted model and random forest model. The k-fold cross-validation results, with a mean RMSE of 70.75 and a standard deviation of 39.69, indicate that the model's predictive accuracy and consistency are acceptable. The moderate variability reflected in the standard deviation suggests that the model performs reliably across different data subsets. Overall, the model is effective and stable, providing satisfactory predictive performance for the given dataset.</p>

# <h2><center>Making Predictions</center></h2>

# In[1391]:


# Monthly average temperatures for Independence, MO (in °F)
monthly_avg_temps = {
    1: 30.2, 2: 35.2, 3: 45.0, 4: 56.0, 5: 66.4, 6: 75.8,
    7: 79.4, 8: 77.9, 9: 70.6, 10: 58.7, 11: 46.6, 12: 34.0
}

# Function to get the simulated current temperature based on the date
def get_simulated_temperature(date):
    avg_temp = monthly_avg_temps[date.month]
    return np.random.uniform(avg_temp - 10, avg_temp + 10)

# Get the current date
current_date = datetime.now()

# Random date for testing different temp ranges
# Generate a random date within a specified range
start_date = datetime(2023, 7, 1)
end_date = datetime(2024, 7, 1)
random_date = start_date + timedelta(days=random.randint(0, (end_date - start_date).days))

# Get the simulated current temperature
current_temperature = get_simulated_temperature(current_date)
random_temperature = get_simulated_temperature(random_date)

# Extract features from the current date
current_month = current_date.month
current_day_of_week = current_date.weekday() + 1  # weekday() returns 0 for Monday, so adding 1
current_day_of_year = current_date.timetuple().tm_yday
current_week_of_year = current_date.isocalendar()[1]

# Extract features from the random date
random_month = random_date.month
random_day_of_week = random_date.weekday() + 1  # weekday() returns 0 for Monday, so adding 1
random_day_of_year = random_date.timetuple().tm_yday
random_week_of_year = random_date.isocalendar()[1]

# Calculate temperature difference feature for the current date
previous_day = current_date - timedelta(days=1)
previous_day_temp = df[df['date'] == previous_day.strftime('%Y-%m-%d')]['temperature'].values
if len(previous_day_temp) > 0:
    temp_diff = current_temperature - previous_day_temp[0]
else:
    temp_diff = 0  # Default to 0 if previous day temperature is not available

# Calculate rolling average feature (using a 7-day window as an example)
rolling_window = 7
rolling_avg_temp = df['temperature'].rolling(window=rolling_window).mean().iloc[-1]

# Ensure the input data has the same number of features as the training data
feature_names = X_train.columns.drop('date')  # Extract the feature names used in training, excluding 'date'

# Prepare the input data for prediction for the current date
current_user_input_data = pd.DataFrame(np.array([[
    current_temperature, 
    temp_diff, 
    rolling_avg_temp, 
    current_month, 
    current_day_of_week, 
    current_day_of_year, 
    current_week_of_year
]]), columns=feature_names[:7])

# Add placeholders for any additional features to match the feature count for the current date
for col in feature_names[7:]:
    current_user_input_data[col] = 0

# Make a prediction using the best linear regression model for the current date
current_user_prediction = best_linear_model.predict(current_user_input_data[feature_names])

# Prepare the input data for prediction for the random date
random_user_input_data = pd.DataFrame(np.array([[
    random_temperature, 
    temp_diff,  # Assuming temperature difference is the same for simplicity
    rolling_avg_temp,  # Assuming rolling average is the same for simplicity
    random_month, 
    random_day_of_week, 
    random_day_of_year, 
    random_week_of_year
]]), columns=feature_names[:7])

# Add placeholders for any additional features to match the feature count for the random date
for col in feature_names[7:]:
    random_user_input_data[col] = 0

# Make a prediction using the best linear regression model for the random date
random_user_prediction = best_linear_model.predict(random_user_input_data[feature_names])

# Display the current prediction
print(f"Current date: {current_date.strftime('%Y-%m-%d')}")
print(f"Simulated current temperature in Independence, MO: {current_temperature}°F")
print(f"Predicted revenue for a temperature of {current_temperature}°F on {current_date.strftime('%Y-%m-%d')} is: ${current_user_prediction[0]:.2f}")

print(f"")

# Display the randomized prediction
print(f"Random date: {random_date.strftime('%Y-%m-%d')}")
print(f"Simulated temperature in Independence, MO: {random_temperature}°F")
print(f"Predicted revenue for a temperature of {random_temperature}°F on {random_date.strftime('%Y-%m-%d')} is: ${random_user_prediction[0]:.2f}")


# In[ ]:




