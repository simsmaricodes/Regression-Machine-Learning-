#!/usr/bin/env python
# coding: utf-8

# ## Case Challenge Part I (Individual Assignment 1)
# After three years serving customers across the San Francisco Bay Area, the executives at
# Apprentice Chef have decided to take on an analytics project to better understand how much
# revenue to expect from each customer within their first year of using their services. Thus, they
# have hired you on a full-time contract to analyze their data, develop your top insights, and build a
# machine learning model to predict revenue over the first year of each customer’s life cycle. They
# have explained to you that for this project, they are not interested in a time series analysis and
# instead would like to “keep things simple” by providing you with a dataset of aggregated
# customer information.

# ## Part 1: Data Exploration
# <h3> Package imports, peaking into data and checking for missing values

# In[1]:


# Importing libraries
# Importing libraries
import pandas as pd  # Data science essentials
import matplotlib.pyplot as plt  # Essential graphical output
import seaborn as sns  # Enhanced graphical output
import numpy as np  # Mathematical essentials
import statsmodels.formula.api as smf  # Regression modeling
from os import listdir  # Look inside file directory
from sklearn.model_selection import train_test_split  # Split data into training and testing data
import gender_guesser.detector as gender  # Guess gender based on (given) name
from sklearn.linear_model import LinearRegression  # OLS Regression
import sklearn.linear_model  # Linear models
from sklearn.neighbors import KNeighborsRegressor  # KNN for Regression
from sklearn.preprocessing import StandardScaler  # standard scaler
import openpyxl

# setting pandas print options
pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Filepath
file = './Apprentice_Chef_Dataset.xlsx'

# Importing the dataset
apprentice = pd.read_excel(io=file)

# formatting and printing the dimensions of the dataset
print(f"""
Size of Original Dataset
------------------------
Observations: {apprentice.shape[0]}
Features:     {apprentice.shape[1]}

There are {apprentice.isnull().any().sum()} missing values 
""")

# In[2]:


# Look at the data
apprentice.head()

# In[3]:


# Checking for missing values
apprentice.isnull().any()

# The missing value is in Family name, which will not be used

# <hr style="height:.9px;border:none;color:#333;background-color:#333;" /><br>
# <h3>Analyzing the Distribution of Revenues</h3>
# <h4>Develop a histogram to analyze the distribution of the Y-variable.</h4>

# In[4]:


# Histogram to check distribution of the response variable
sns.displot(data=apprentice,
            x='REVENUE',
            height=5,
            aspect=2)

# displaying the histogram
plt.show()

# <h4>Develop a histogram to analyze the distribution of the log of the Y-variable.</h4>

# In[5]:


# log transforming Sale_Price and saving it to the dataset
apprentice['log_REVENUE'] = np.log10(apprentice['REVENUE'])

# developing a histogram using for log Revenue
sns.displot(data=apprentice,
            x='log_REVENUE',
            height=5,
            aspect=2)

# displaying the histogram
plt.show()

# The log data is a bit better although there is still that under represented data point

# <hr style="height:.9px;border:none;color:#333;background-color:#333;" /><br>
#
# <h3>Based on the outputs above, identify the data type of each original variable in the dataset.</h3><br>
# Use the following groupings:
#
# * CONTINUOUS
# * INTERVAL/COUNT
# * CATEGORICAL


# <hr style="height:.9px;border:none;color:#333;background-color:#333;" /><br>
#
# ## Part 2: Trend Based Features
# <h3>Checking the Continuous Data</h3>

# In[6]:


########################
# Visual EDA (Scatterplots)
########################

# setting figure size
fig, ax = plt.subplots(figsize=(10, 8))

# developing a scatterplot
plt.subplot(2, 2, 1)
sns.scatterplot(x=apprentice['AVG_TIME_PER_SITE_VISIT'],
                y=apprentice['REVENUE'],
                color='g')

# adding labels but not adding title
plt.xlabel(xlabel='Average Visit Time')
plt.ylabel(ylabel='Revenue')

########################


# developing a scatterplot
plt.subplot(2, 2, 2)
sns.scatterplot(x=apprentice['AVG_PREP_VID_TIME'],
                y=apprentice['REVENUE'],
                color='g')

# adding labels but not adding title
plt.xlabel(xlabel='Average Video Time')
plt.ylabel(ylabel='Revenue')

########################


# developing a scatterplot
plt.subplot(2, 2, 3)
sns.scatterplot(x=apprentice['TOTAL_PHOTOS_VIEWED'],
                y=apprentice['REVENUE'],
                color='orange')

# adding labels but not adding title
plt.xlabel(xlabel='Totals Meals')
plt.ylabel(ylabel='Revenue')

########################


# developing a scatterplot
plt.subplot(2, 2, 4)
sns.scatterplot(x=apprentice['TOTAL_MEALS_ORDERED'],
                y=apprentice['REVENUE'],
                color='r')

# adding labels but not adding title
plt.xlabel(xlabel='Total Meals')
plt.ylabel(ylabel='Revenue')

# cleaning up the layout and displaying the results
plt.tight_layout()
plt.show()

# It is clear that from the data collection method the Median Meal Rating and Average clicks per visit can be counted in Count data as they are not continuous data

# <h3>Checking the Interval and Count Data</h3>

# In[7]:


# Counting the number of zeroes in the interval data

noon_canc_zeroes = apprentice['CANCELLATIONS_BEFORE_NOON'].value_counts()[0]
after_canc_zeroes = apprentice['CANCELLATIONS_AFTER_NOON'].value_counts()[0]
weekly_log_zeroes = apprentice['WEEKLY_PLAN'].value_counts()[0]
early_meal_zeroes = apprentice['EARLY_DELIVERIES'].value_counts()[0]
late_meal_zeroes = apprentice['LATE_DELIVERIES'].value_counts()[0]
master_class_zeroes = apprentice['MASTER_CLASSES_ATTENDED'].value_counts()[0]
photo_view = apprentice['TOTAL_PHOTOS_VIEWED'].value_counts()[0]

# printing a table of the results
print(f"""
                               Yes\t\tNo
                              ---------------------
Cancellations Before Noon    | {noon_canc_zeroes}\t\t{len(apprentice) - noon_canc_zeroes}
Cancellations After Noon     | {after_canc_zeroes}\t\t{len(apprentice) - after_canc_zeroes}
Weekly plan Subscription     | {weekly_log_zeroes}\t\t{len(apprentice) - weekly_log_zeroes}
Early Meals.                 | {early_meal_zeroes}\t\t{len(apprentice) - early_meal_zeroes}
Late Meals.                  | {late_meal_zeroes}\t\t{len(apprentice) - late_meal_zeroes}
Master Class Attendance      | {master_class_zeroes}\t\t{len(apprentice) - master_class_zeroes}
Photo Views.                 | {photo_view}\t\t{len(apprentice) - photo_view}
""")

# In[8]:


# Dummy Variables for the factors we found above with at leasst 100 observations
apprentice['noon_canc'] = 0
apprentice['after_canc'] = 0
apprentice['weekly_plan_sub'] = 0
apprentice['early_delivery'] = 0
apprentice['late_delivery'] = 0
apprentice['masterclass_att'] = 0
apprentice['view_photo'] = 0

# Iter over eachg column to get the new boolean feature columns
for index, value in apprentice.iterrows():

    # For noon cancellations
    if apprentice.loc[index, 'CANCELLATIONS_BEFORE_NOON'] > 0:
        apprentice.loc[index, 'noon_canc'] = 1

    # For afternoon cancelations
    if apprentice.loc[index, 'CANCELLATIONS_AFTER_NOON'] > 0:
        apprentice.loc[index, 'after_canc'] = 1

    # Weekly meal plan subscription
    if apprentice.loc[index, 'WEEKLY_PLAN'] > 0:
        apprentice.loc[index, 'weekly_plan_sub'] = 1

    # Early deliveries
    if apprentice.loc[index, 'EARLY_DELIVERIES'] > 0:
        apprentice.loc[index, 'early_delivery'] = 1

    # Late Deliveries
    if apprentice.loc[index, 'LATE_DELIVERIES'] > 0:
        apprentice.loc[index, 'late_delivery'] = 1

    # Masterclass attendance
    if apprentice.loc[index, 'MASTER_CLASSES_ATTENDED'] > 0:
        apprentice.loc[index, 'masterclass_att'] = 1

    # Viewed Photos
    if apprentice.loc[index, 'TOTAL_PHOTOS_VIEWED'] > 0:
        apprentice.loc[index, 'view_photo'] = 1

# Another Factor i want to consider is make flags for whether the customer contacted customer services on more than half of their orders and whether the mobile or pc is the preffered route of ordering.

# In[9]:


# Checking distribution
contact_greater = []
mobile_greater = []

# Instantiating dummy variables
for index, value in apprentice.iterrows():

    # For noon cancellations
    if apprentice.loc[index, 'CONTACTS_W_CUSTOMER_SERVICE'] > (apprentice.loc[index, 'TOTAL_MEALS_ORDERED']) / 2:
        contact_greater.append(1)
    else:
        contact_greater.append(0)

# Instantiating dummy variables
for index, value in apprentice.iterrows():

    if apprentice.loc[index, 'MOBILE_LOGINS'] > apprentice.loc[index, 'PC_LOGINS']:
        mobile_greater.append(1)

    else:
        mobile_greater.append(0)

contact_greater = pd.DataFrame(contact_greater)
mobile_greater = pd.DataFrame(mobile_greater)  # PC logins are consistently more so we dop

contact_greater.value_counts()  # Checking distribution of zeros

# Adding them to the data
apprentice['contact_greater'] = contact_greater
apprentice['mobile_greater'] = mobile_greater

# In[10]:


# <h4>Checking the Count and interval data after dealing with zeroes</h4>
# Some of the count data had significant information in zeroes to split them into some sort of boolean feature. Now, I will plot to distributions of interval to see which data might need transformation to give insight into our model.


# After checking the plots for all the interval data these were the ones needing transformation.

# In[11]:


# setting figure size
fig, ax = plt.subplots(figsize=(15, 10))

## Plot 1: Original X, Original Y ##
plt.subplot(1, 2, 1)
# Plotting
sns.boxplot(x='AVG_CLICKS_PER_VISIT',
            y='REVENUE',
            data=apprentice
            )

# titles and labels
plt.title('Average clicks per visit')

## Plot 1: Original X, Original Y ##

plt.subplot(1, 2, 2)
# Plotting
sns.boxplot(x='CONTACTS_W_CUSTOMER_SERVICE',
            y='REVENUE',
            data=apprentice
            )

# titles and labels
plt.title('Customer Service')

# Showing the displaying
plt.show()

# In[12]:


# Converting to logs and seeing if the data improves
apprentice['log_clicks'] = np.log10(apprentice['AVG_CLICKS_PER_VISIT'])  # Average clicks log
apprentice['log_customer'] = np.log10(apprentice['CONTACTS_W_CUSTOMER_SERVICE'])  # Customer contact

# setting figure size
fig, ax = plt.subplots(figsize=(15, 10))

## Plot 1: Original X, Original Y ##
plt.subplot(1, 2, 1)
# Plotting
sns.boxplot(x='log_clicks',
            y='log_REVENUE',
            data=apprentice
            )

# titles and labels
plt.title('LOG Average clicks per visit')

## Plot 1: Original X, Original Y ##

plt.subplot(1, 2, 2)
# Plotting
sns.boxplot(x='log_customer',
            y='log_REVENUE',
            data=apprentice
            )

# titles and labels
plt.title('LOG Customer Service')

# Showing the displaying
plt.show()

# In[13]:


# Dummy Variables for the factors we found above with at leasst 100 observations
apprentice['meals_below_fif'] = 0
apprentice['meals_above_two'] = 0
apprentice['unique_meals_above_ten'] = 0
apprentice['cust_serv_under_ten'] = 0
apprentice['click_under_eight'] = 0

# Iter over eachg column to get the new boolean feature columns

for index, value in apprentice.iterrows():

    # Total meals greater than 200
    if apprentice.loc[index, 'TOTAL_MEALS_ORDERED'] >= 200:
        apprentice.loc[index, 'meals_below_fif'] = 1

    # Total meals less than 15
    if apprentice.loc[index, 'TOTAL_MEALS_ORDERED'] <= 15:
        apprentice.loc[index, 'meals_above_two'] = 1

    # Unique meals greater 10
    if apprentice.loc[index, 'UNIQUE_MEALS_PURCH'] > 10:
        apprentice.loc[index, 'unique_meals_above_ten'] = 1

    # Customer service less than 10
    if apprentice.loc[index, 'CONTACTS_W_CUSTOMER_SERVICE'] < 10:
        apprentice.loc[index, 'cust_serv_under_ten'] = 1

    # Clicks below 8
    if apprentice.loc[index, 'AVG_CLICKS_PER_VISIT'] < 8:
        apprentice.loc[index, 'click_under_eight'] = 1

# Adding the new variable
apprentice['freq_customer_service'] = 0

# Instantiating dummy variables
for index, value in apprentice.iterrows():

    # For noon cancellations
    if apprentice.loc[index, 'CONTACTS_W_CUSTOMER_SERVICE'] > (apprentice.loc[index, 'TOTAL_MEALS_ORDERED']) / 2:
        apprentice.loc[index, 'freq_customer_service'] = 1

# In[14]:


# Log transforms

inter_list = ['LARGEST_ORDER_SIZE', 'PRODUCT_CATEGORIES_VIEWED', 'PC_LOGINS',
              'TOTAL_MEALS_ORDERED', 'UNIQUE_MEALS_PURCH', 'CONTACTS_W_CUSTOMER_SERVICE']

for item in inter_list:
    # Converting to logs and seeing if the data improves
    apprentice['log_' + item] = np.log10(apprentice[item])

# <h3>Working with Categorical Data</h3>

# In[15]:


# STEP 1: splitting personal emails

# placeholder list
placeholder_lst = []

# looping over each email address
for index, col in apprentice.iterrows():
    # splitting email domain at '@'
    split_email = apprentice.loc[index, 'EMAIL'].split(sep='@')

    # appending placeholder_lst with the results
    placeholder_lst.append(split_email)

# converting placeholder_lst into a DataFrame
email_df = pd.DataFrame(placeholder_lst)

# STEP 2: concatenating with original DataFrame
# renaming column to concatenate
email_df.columns = ['0', 'personal_email_domain']

# concatenating personal_email_domain with friends DataFrame
apprentice = pd.concat([apprentice, email_df['personal_email_domain']],
                       axis=1)

# In[16]:


# printing value counts of personal_email_domain
apprentice.loc[:, 'personal_email_domain'].value_counts()

# In[17]:


# email domain types
personal_email_domains = ['@gmail.com', '@microsoft.com', '@yahoo.com',
                          '@msn.com', '@live.com', '@protonmail.com',
                          '@aol.com', '@hotmail.com', '@apple.com']

# Domain list
domain_lst = []

# looping to group observations by domain type
for domain in apprentice['personal_email_domain']:
    if '@' + domain in personal_email_domains:
        domain_lst.append('personal')

    else:
        domain_lst.append('work')

# concatenating with original DataFrame
apprentice['domain_group'] = pd.Series(domain_lst)

# checking results
apprentice['domain_group'].value_counts()

# Created some extra categorical data that we can use to try infer some more statistics

# In[18]:


# one hot encoding categorical variables
one_hot_domain = pd.get_dummies(apprentice['domain_group'])

# joining codings together
apprentice = apprentice.join([one_hot_domain])

# In[19]:


apprentice.describe()

# <hr style="height:.9px;border:none;color:#333;background-color:#333;" /><br>
#
# ## Part 3: Model Testing
# <br>

# In[20]:


# making a copy of housing
apprentice_explanatory = apprentice.copy()

# dropping SalePrice and Order from the explanatory variable set
apprentice_explanatory = apprentice_explanatory.drop(['REVENUE', 'NAME', 'EMAIL', 'FIRST_NAME',
                                                      'FAMILY_NAME', 'personal_email_domain', 'domain_group',
                                                      'log_REVENUE'], axis=1)

# formatting each explanatory variable for statsmodels
for val in apprentice_explanatory:
    print(val, '+')

# In[21]:


# Step 1: build a model
lm_best = smf.ols(formula="""log_REVENUE ~ CROSS_SELL_SUCCESS +
                                                UNIQUE_MEALS_PURCH +
                                                CONTACTS_W_CUSTOMER_SERVICE +
                                                PRODUCT_CATEGORIES_VIEWED +
                                                AVG_PREP_VID_TIME +
                                                LARGEST_ORDER_SIZE +
                                                MEDIAN_MEAL_RATING +
                                                AVG_CLICKS_PER_VISIT +
                                                masterclass_att +
                                                view_photo +
                                                contact_greater +
                                                mobile_greater +
                                                log_clicks +
                                                log_customer +
                                                meals_below_fif +
                                                meals_above_two +
                                                unique_meals_above_ten +
                                                click_under_eight +
                                                freq_customer_service +
                                                log_LARGEST_ORDER_SIZE +
                                                log_PRODUCT_CATEGORIES_VIEWED +
                                                log_TOTAL_MEALS_ORDERED +
                                                log_UNIQUE_MEALS_PURCH +
                                                log_CONTACTS_W_CUSTOMER_SERVICE +
                                                personal +
                                                work """,
                  data=apprentice)

# Step 2: fit the model based on the data
results = lm_best.fit()

# Step 3: analyze the summary output
print(results.summary())

# In[22]:


# preparing explanatory variable data

x_variables = ['CROSS_SELL_SUCCESS', 'UNIQUE_MEALS_PURCH', 'CONTACTS_W_CUSTOMER_SERVICE',
               'PRODUCT_CATEGORIES_VIEWED', 'AVG_PREP_VID_TIME', 'LARGEST_ORDER_SIZE',
               'MEDIAN_MEAL_RATING', 'AVG_CLICKS_PER_VISIT', 'masterclass_att',
               'view_photo', 'log_clicks', 'log_customer', 'meals_below_fif',
               'meals_above_two', 'unique_meals_above_ten', 'click_under_eight',
               'freq_customer_service', 'log_LARGEST_ORDER_SIZE', 'log_PRODUCT_CATEGORIES_VIEWED',
               'log_TOTAL_MEALS_ORDERED', 'log_UNIQUE_MEALS_PURCH', 'log_CONTACTS_W_CUSTOMER_SERVICE',
               'personal', 'work']

apprentice_data = apprentice_explanatory[x_variables]

# preparing the target variable
apprentice_target = apprentice.loc[:, 'log_REVENUE']

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(
    apprentice_data,
    apprentice_target,
    test_size=0.25,
    random_state=219)

# In[23]:


# INSTANTIATING a model object
lr = LinearRegression()

# FITTING to the training data
lr_fit = lr.fit(X_train, y_train)

# PREDICTING on new data
lr_pred = lr_fit.predict(X_test)

# SCORING the results
print('OLS Training Score :', lr.score(X_train, y_train).round(4))  # using R-square
print('OLS Testing Score  :', lr.score(X_test, y_test).round(4))  # using R-square

lr_train_score = lr.score(X_train, y_train).round(4)
lr_test_score = lr.score(X_test, y_test).round(4)

# displaying and saving the gap between training and testing
print('OLS Train-Test Gap :', abs(lr_train_score - lr_test_score).round(4))
lr_test_gap = abs(lr_train_score - lr_test_score).round(4)

# In[24]:


# zipping each feature name to its coefficient
lr_model_values = zip(apprentice_data.columns,
                      lr_fit.coef_.round(decimals=4))

# setting up a placeholder list to store model features
lr_model_lst = [('intercept', lr_fit.intercept_.round(decimals=4))]

# printing out each feature-coefficient pair one by one
for val in lr_model_values:
    lr_model_lst.append(val)

# checking the results
for pair in lr_model_lst:
    print(pair)

# In[25]:


# Making the list a data frame to print later
lr_model_lst = pd.DataFrame(lr_model_lst)

# Naming the Columns
lr_model_lst.columns = ['Variables', 'Coefficients']

# Removing indices for print
lr_model_lst_no_indices = lr_model_lst.to_string(index=False)

# In[26]:


# Importing another library
import sklearn.linear_model  # Linear models

# In[27]:


# INSTANTIATING a model object
lasso_model = sklearn.linear_model.Lasso()  # default magitude

# FITTING to the training data
lasso_fit = lasso_model.fit(X_train, y_train)

# PREDICTING on new data
lasso_pred = lasso_fit.predict(X_test)

# SCORING the results
print('Lasso Training Score :', lasso_model.score(X_train, y_train).round(4))
print('Lasso Testing Score  :', lasso_model.score(X_test, y_test).round(4))

## the following code has been provided for you ##

# saving scoring data for future use
lasso_train_score = lasso_model.score(X_train, y_train).round(4)  # using R-square
lasso_test_score = lasso_model.score(X_test, y_test).round(4)  # using R-square

# displaying and saving the gap between training and testing
print('Lasso Train-Test Gap :', abs(lr_train_score - lr_test_score).round(4))
lasso_test_gap = abs(lr_train_score - lr_test_score).round(4)

# In[28]:


# zipping each feature name to its coefficient
lasso_model_values = zip(apprentice_data.columns, lasso_fit.coef_.round(decimals=2))

# setting up a placeholder list to store model features
lasso_model_lst = [('intercept', lasso_fit.intercept_.round(decimals=2))]

# printing out each feature-coefficient pair one by one
for val in lasso_model_values:
    lasso_model_lst.append(val)

# checking the results
for pair in lasso_model_lst:
    print(pair)

# In[29]:


# INSTANTIATING a model object
ard_model = sklearn.linear_model.ARDRegression()

# FITTING the training data
ard_fit = ard_model.fit(X_train, y_train)

# PREDICTING on new data
ard_pred = ard_fit.predict(X_test)

print('ARD Training Score:', ard_model.score(X_train, y_train).round(4))
print('ARD Testing Score :', ard_model.score(X_test, y_test).round(4))

# saving scoring data for future use
ard_train_score = ard_model.score(X_train, y_train).round(4)
ard_test_score = ard_model.score(X_test, y_test).round(4)

# displaying and saving the gap between training and testing
print('ARD Train-Test Gap :', abs(ard_train_score - ard_test_score).round(4))
ard_test_gap = abs(ard_train_score - ard_test_score).round(4)

# In[30]:


# zipping each feature name to its coefficient
ard_model_values = zip(apprentice_data.columns, ard_fit.coef_.round(decimals=5))

# setting up a placeholder list to store model features
ard_model_lst = [('intercept', ard_fit.intercept_.round(decimals=2))]

# printing out each feature-coefficient pair one by one
for val in ard_model_values:
    ard_model_lst.append(val)

# checking the results
for pair in ard_model_lst:
    print(pair)

# In[31]:


# KNN
# INSTANTIATING a StandardScaler() object
scaler = StandardScaler()

# FITTING the scaler with the data
scaler.fit(apprentice_data)

# TRANSFORMING our data after fit
X_scaled = scaler.transform(apprentice_data)

# converting scaled data into a DataFrame
X_scaled_df = pd.DataFrame(X_scaled)

# adding labels to the scaled DataFrame
X_scaled_df.columns = apprentice_data.columns

# Training testing and splitit again
X_train_STAND, X_test_STAND, y_train_STAND, y_test_STAND = train_test_split(
    X_scaled_df,
    apprentice_target,
    test_size=0.25,
    random_state=219)

# INSTANTIATING a model with the optimal number of neighbors
knn_stand = KNeighborsRegressor(algorithm='auto',
                                n_neighbors=9)

# FITTING the model based on the training data
knn_stand_fit = knn_stand.fit(X_train_STAND, y_train_STAND)

# PREDITCING on new data
knn_stand_pred = knn_stand_fit.predict(X_test)

# SCORING the results
print('KNN Training Score:', knn_stand.score(X_train_STAND, y_train_STAND).round(4))
print('KNN Testing Score :', knn_stand.score(X_test_STAND, y_test_STAND).round(4))

# saving scoring data for future use
knn_stand_score_train = knn_stand.score(X_train_STAND, y_train_STAND).round(4)
knn_stand_score_test = knn_stand.score(X_test_STAND, y_test_STAND).round(4)

# displaying and saving the gap between training and testing
print('KNN Train-Test Gap:', abs(knn_stand_score_train - knn_stand_score_test).round(4))
knn_stand_test_gap = abs(knn_stand_score_train - knn_stand_score_test).round(4)

# In[32]:


# comparing results

print(f"""
Model      Train Score      Test Score      Train-Test Gap     Model Size
-----      -----------      ----------      ---------------    ----------
OLS        {lr_train_score}            {lr_test_score}          {lr_test_gap}               {len(lr_model_lst)}
Lasso      {lasso_train_score}           {lasso_test_score}          {lasso_test_gap}               {len(lasso_model_lst)}
ARD        {ard_train_score}           {ard_test_score}          {ard_test_gap}               {len(ard_model_lst)}
""")

# In[33]:


# creating a dictionary for model results
model_performance = {

    'Model Type': ['OLS', 'Lasso', 'ARD'],

    'Training': [lr_train_score, lasso_train_score,
                 ard_train_score],

    'Testing': [lr_test_score, lasso_test_score,
                ard_test_score],

    'Train-Test Gap': [lr_test_gap, lasso_test_gap,
                       ard_test_gap],

    'Model Size': [len(lr_model_lst), len(lasso_model_lst),
                   len(ard_model_lst)],

    'Model': [lr_model_lst, lasso_model_lst, ard_model_lst]}

# converting model_performance into a DataFrame
model_performance = pd.DataFrame(model_performance)

model_performance.head()

# <hr style="height:.9px;border:none;color:#333;background-color:#333;" /><br>
#
# ## Part 4: Final Model Selected
#
# The best model from the above analysis is the OLS regression which has the following:
#

# In[34]:


#  Selected Model
print(f"""

The Model selected is OLS Regression 

Model      Train Score      Test Score      Train-Test Gap     Model Size
-----      -----------      ----------      ---------------    ----------
OLS        {lr_train_score}            {lr_test_score}          {lr_test_gap}               {len(lr_model_lst)}


Model Coefficients
----------------------

{lr_model_lst_no_indices}


""")



