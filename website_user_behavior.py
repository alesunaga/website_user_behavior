# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Read in the data
website = pd.read_csv('website.csv')

# Print the first five rows
print(website.head())

# Create a scatter plot of time vs age
plt.scatter(website.age, website.time_seconds)
plt.xlabel("Age")
plt.ylabel("Time (seconds)")
plt.title("Time Spent on Website vs. Age")
plt.show()
plt.clf()  # Clear the plot

# Fit a linear regression to predict time_seconds based on age
model = sm.OLS.from_formula('time_seconds ~ age', website)
results = model.fit()
print(results.params)

# Plot the scatter plot with the regression line
plt.scatter(website.age, website.time_seconds)
plt.xlabel("Age")
plt.ylabel("Time (seconds)")
plt.title("Time Spent on Website vs. Age")
predicted_time = results.params[1] * website.age + results.params[0]
plt.plot(website.age, predicted_time, color='red', label='Regression Line')
plt.legend()
plt.show()
plt.clf()

# Calculate fitted values and residuals
fitted_values = results.predict(website.age)
residuals = website.time_seconds - fitted_values

# Diagnostic plots
plt.hist(residuals)
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Histogram of Residuals")
plt.show()
plt.clf()

plt.scatter(fitted_values, residuals)
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs. Fitted Values")
plt.show()
plt.clf()

# Predict amount of time on website for a 40 year old
pred_40yr = results.params[1] * 40 + results.params[0]
print("Predicted time for a 40 year old:", pred_40yr)

# Analyze time spent based on browser
print(website.groupby('browser').mean().time_seconds)

# Bar plot of average time spent by browser
mean_time_Chrome = np.mean(website.time_seconds[website.browser == 'Chrome'])
mean_time_Safari = np.mean(website.time_seconds[website.browser == 'Safari'])

plt.bar(['Chrome', 'Safari'], [mean_time_Chrome, mean_time_Safari])
plt.xlabel("Browser")
plt.ylabel("Average Time (seconds)")
plt.title("Average Time Spent by Browser")
plt.show()
plt.clf()

# Fit a linear regression to predict time_seconds based on browser
model = sm.OLS.from_formula('time_seconds ~ browser', website)
results = model.fit()
print(results.params)
