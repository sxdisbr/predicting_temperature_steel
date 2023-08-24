#!/usr/bin/env python
# coding: utf-8

# ### Description and purpose of the study
# 
# In order to optimize production costs, a metallurgical factory decided to reduce electricity consumption at the steel processing stage. We have to build a model that predicts the temperature of steel.

# ### Description of the processing stage
# 
# Steel is processed in a metal bucket with a capacity of about 100 tons. In order for the bucket to withstand high temperatures, it is lined with refractory bricks from the inside. The molten steel is poured into a ladle and heated to the desired temperature with graphite electrodes. They are installed in the bucket lid.
# 
# Sulfur is removed from the alloy (desulfurization), the chemical composition is corrected by adding impurities and samples are taken. Steel is alloyed — its composition is changed — by feeding alloy pieces from a hopper for bulk materials or wire through a special tribe apparatus (English tribe, "mass").
# 
# Before the first time alloying additives are introduced, the temperature of the steel is measured and its chemical analysis is performed. Then the temperature is increased for a few minutes, alloying materials are added and the alloy is purged with an inert gas. Then it is mixed and measurements are carried out again. This cycle is repeated until the target chemical composition and optimal melting temperature are reached.
# 
# Then the molten steel is sent to the metal finishing or enters the continuous casting machine. From there, the finished product comes out in the form of slabs.

# ## Data Description
# 
# The data consists of files received from different sources:
# 
# data_arc_new.csv — data about electrodes;
# 
# data_bulk_new.csv — data on the supply of bulk materials (volume);
# 
# data_bulk_time_new.csv — data on the supply of bulk materials (time);
# 
# data_gas_new.csv — data on gas purging of the alloy;
# 
# data_temp_new.csv — temperature measurement results;
# 
# data_wire_new.csv — data on wire materials (volume);
# 
# data_wire_time_new.csv — data about wire materials (time).

# ### Work plan
# 
# - Data preprocessing. Observation of data types, variables, and data processing.
# 
# - Development of functions. Construction of variables of interest to us with special emphasis on variables related to temperature.
# 
# - As soon as the data is processed and the variables are processed, we will create a table combining all the columns of interest. We will divide between features and target, which in our case is the temperature.
# 
# - We will divide the data set for verification in a ratio of 3:1.
# 
# - Prepare all the columns and divide the main functions into target ones.
# 
# - We will select a suitable regression model. Among the regressive models that could work well, we consider CatBoost and LightBoost.
# 
# - We will evaluate the effectiveness of regression models in order to then choose the best model and submit it to the testing stage. Our goal is to achieve a lower score equal to or less than 6.8.

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from catboost import CatBoostRegressor 
from sklearn.model_selection import cross_val_score 
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor


# In[2]:


data_arc = pd.read_csv(r'C:\Users\pinos\Downloads/data_arc_new.csv')
data_bulk = pd.read_csv(r'C:\Users\pinos\Downloads/data_bulk_new.csv')
data_bulk_time = pd.read_csv(r'C:\Users\pinos\Downloads/data_bulk_time_new.csv', index_col=0)
data_gas = pd.read_csv(r'C:\Users\pinos\Downloads/data_gas_new.csv')
data_temp = pd.read_csv(r'C:\Users\pinos\Downloads/data_temp_new.csv')
data_wire = pd.read_csv(r'C:\Users\pinos\Downloads/data_wire_new.csv')
data_wire_time = pd.read_csv(r'C:\Users\pinos\Downloads/data_wire_time_new.csv')


# In[3]:


data_arc.head()


# In[4]:


plt.figure(figsize=(20,10))
plt.hist(data_arc['Активная мощность'], bins=100)
plt.title('Активная мощность')
plt.xlabel('Мощность')
plt.ylabel('Частота')
plt.show()


# The data in this column looks fine.

# In[5]:


data_arc['Активная мощность'].describe()


# In[6]:


plt.figure(figsize=(20,10))
plt.hist(data_arc['Реактивная мощность'], bins=100)
plt.title('Реактивная мощность')
plt.xlabel('Мощность')
plt.ylabel('Частота')
plt.show()


# Something strange is happening here, so we will consider the maximum and minimum.

# In[7]:


data_arc['Реактивная мощность'].max()


# In[8]:


data_arc['Реактивная мощность'].min()


# Effectively we will remove atypical values less than 0.

# In[9]:


data_arc = data_arc[data_arc['Реактивная мощность'] > 0]


# In[10]:


plt.figure(figsize=(20,10))
plt.hist(data_arc['Реактивная мощность'], bins=100)
plt.title('Реактивная мощность')
plt.xlabel('Мощность')
plt.ylabel('Частота')
plt.show()


# Now we see the values as normal parameters.

# In[11]:


data_arc['Реактивная мощность'].describe()


# The next thing we're going to do with this dataset is change the time to datetime.

# In[12]:


data_arc['Начало нагрева дугой'] = pd.to_datetime(data_arc['Начало нагрева дугой'] )


# In[13]:


data_arc['Конец нагрева дугой'] = pd.to_datetime(data_arc['Конец нагрева дугой'] )


# In[14]:


data_arc['electrode_duration'] = data_arc.groupby('key')['Конец нагрева дугой'].max() - data_arc.groupby('key')['Начало нагрева дугой'].min()


# In[15]:


# We check that the changes were made correctly
data_arc.head()


# ### Conclusion
# 
# In this first data set, we find the heating time, both initial and final, in addition to its active and reactive power. The active power values are distributed between 0.25 and 1.4, while most of the data is distributed between 0.3 and 1. The reactive power has values from about 0.1 to 1.2, while the bulk of the data is distributed between 0.3 and 0.6. We also remove static values less than zero.

# In[16]:


data_bulk = pd.read_csv(r'/datasets/data_bulk_new.csv')


# In[17]:


data_bulk.head()


# In[18]:


data_bulk.info()


# In[19]:


plt.figure(figsize=(20,10))
plt.hist(data_bulk)
plt.title('Сырье для плавки')
plt.xlabel('Количество сырья')
plt.ylabel('Частота')
plt.show()


# We are running a loop to see the variables and their visual representation more clearly.

# In[20]:


for column in data_bulk.columns:

    data_bulk[column].hist(bins=100, figsize=(20, 10), alpha=0.5)
    
    plt.title('Сырье для плавки')
    
    plt.xlabel('Количество сырья')
    
    plt.ylabel('Частота')
    
    plt.show()


# In[21]:


data_bulk_time.head()


# ### Conclusion
# 
# Judging by the descriptive statistics, everything seems to be in order with the raw materials.
# It is noteworthy that there are very wide peaks.
# Raw materials may exhibit an irregular flow, if we take into account the presented missing values, one solution may be to fill these values with zero.
# As for the raw material handling time, we find a lot of NaN, only four columns seem to be complete.

# In[22]:


data_gas.head()


# In[23]:


plt.figure(figsize=(20,10))
plt.hist(data_gas['Газ 1'], bins=100)
plt.title('Газ, используемый в процессе плавления')
plt.xlabel('Количество газ')
plt.ylabel('Частота')
plt.show()


# In[24]:


data_gas['Газ 1'].describe()


# In[25]:


data_gas['Газ 1'].isna().sum()


# ### Conclusion
# 
# We can assume that with 40 we have atypical values, although out of prudence we are not going to exclude them. Most of the data is in the range from 0 to 30.
# Also, we don't have any missing values in this column.

# In[26]:


data_temp.head()


# In[27]:


data_temp['Время замера'] = pd.to_datetime(data_temp['Время замера'])


# In[28]:


plt.figure(figsize=(20,10))
plt.hist(data_temp['Температура'], bins=100)
plt.title('Температура плавления')
plt.xlabel('Температура')
plt.ylabel('Частота')
plt.show()


# In[29]:


data_temp['Температура'].describe()


# In[30]:


data_temp.info()


# ### Output
# 
# The temperature graph shows the normal distribution with values grouped in the temperature range from 1580 to 1600. In principle, we can say that the values will be easily predictable, which will make our task easier.

# In[31]:


data_wire.head()


# In[32]:


plt.figure(figsize=(20,10))
plt.hist(data_wire['Wire 1'], bins=100)
plt.title('Производство кабеля')
plt.xlabel('Кабель')
plt.ylabel('Частота')
plt.show()


# In[33]:


data_wire['Wire 1'].describe()


# In[34]:


data_wire_time.head()


# In[35]:


data_wire_time.info()


# ### Conclusion
# 
# The data in the wire columns behave similarly to the bulk columns, in addition, they are also consistent with the fact that both have a large number of missing values.

# ## General conclusion of EDA
# 
# During the preliminary analysis of the data, we found certain anomalies: in data_arc, we continue to delete values less than zero. In data_bulk data_wire we find a lot of missing values that may be caused by a lack of material in these processes. One way to solve this problem may be to fill with zeros in such a way that it does not affect the analysis or the construction of a forecasting model.
# The gas and temperature variables are presented as normal parameters without missing values, and their graphical form takes the type of a normal distribution, which is a positive fact that will facilitate the implementation of a good forecast.

# ## Building a model

# We prepare the data as a preliminary step to creating a model. We are going to process all the lost data.

# In[36]:


data_bulk=data_bulk.fillna(0)


# In[37]:


data_bulk.head()


# In[38]:


data_arc.isna().value_counts()


# In[39]:


data_wire=data_wire.fillna(0)


# In[40]:


data_wire.head()


# Now we are going to create a pivot table for summing reactive and active power.

# In[41]:


data_arc_sum = pd.pivot_table(data_arc,
                             values=['Активная мощность','Реактивная мощность'],
                             index='key',
                             aggfunc={'Активная мощность': np.sum,
                                      'Реактивная мощность': np.sum})

data_arc_sum.columns = ['sum_active_power','sum_reactive_power']


# In[42]:


data_arc_sum.head()


# In[43]:


data_temp.head()


# We run a loop to make sure that the initial and final maximum temperatures match their respective key, leaving aside all those that don't match.

# In[44]:


keys = []

for key in list(data_temp['key'].unique()):
    
    try:
        if (data_temp[data_temp['key'] == key]['Время замера'].max() < 
            data_arc[data_arc['key'] == key]['Конец нагрева дугой'].max()): 
            keys.append(key)
            
    except:
        keys.append(key)
        
data_temp = data_temp.query('key not in @keys')


# We guarantee that there is not a single lost value that we have forgotten about.

# In[45]:


data_temp = data_temp.dropna()


# We perform another loop in which we check the unique values. If the unique values are less than two, we sum them up. then we exclude those rows in which the value of the key is equal to the value of the iterator.

# In[46]:


for i in (data_temp['key'].unique()): 
    if (data_temp['key']==i).sum() < 2:
        data_temp = data_temp[data_temp.key != i]


# In[47]:


data_temp_time = data_temp.pivot_table(index=['key'], values=('Температура', 'Время замера'), aggfunc=['first', 'last'])


# In[48]:


data_temp_time.head()


# In[49]:


data_temp_time.columns = ['begin_time', 'begin_temperature', 'end_time', 'end_temperature']


# We generate a table indicating the start time of the temperature and the end time indicating its end temperature.

# In[50]:


data_temp_time.head()


# We set the key column as the index.

# In[51]:


data_bulk = data_bulk.set_index('key')
data_gas = data_gas.set_index('key') 
data_wire = data_wire.set_index('key')


# In[52]:


data_arc


# We combine different tables into one.

# In[53]:


# Perform internal merging of indexes of all data frames
data = pd.concat([data_temp_time, data_arc_sum, data_arc, data_bulk, data_gas, data_wire], axis=1, join='inner')


# In[54]:


# Reset the index of the merged data frame

data = data.reset_index(drop=True)


# In[55]:


data.head()


# In[56]:


data


# In[57]:


# Calculate the heating duration (heat_duration) by subtracting the initial time (begin_time) 
# from the end time (end_time)

data['heat_duration'] = data['end_time'] - data['begin_time']





# In[58]:


data.info()


# We delete columns that have no informational value.

# In[59]:


data = data.drop('begin_time',axis=1)
data = data.drop('end_time',axis=1)
data = data.dropna(subset=['end_temperature'])


# In[60]:


data.head()


# Мы меняем типы данных на правильные.

# In[61]:


data['Газ 1']=data['Газ 1'].astype(int)
data['end_temperature']=data['end_temperature'].astype(int)
data['sum_active_power']=data['sum_active_power'].astype(int)
data['sum_reactive_power']=data['sum_reactive_power'].astype(int)
data['begin_temperature']=data['begin_temperature'].astype(int)
data['heat_duration']=data['heat_duration'].astype(int)
data['electrode_duration']=data['electrode_duration'].astype(int)
for i in range(4,16):
    data[f'Bulk {i}'] = data[f'Bulk {i}'].astype(int)
for i in range(1,10):
    data[f'Wire {i}'] = data[f'Wire {i}'].astype(int)


# In[62]:


data = data.drop(['Bulk 1', 'Bulk 2', 'Bulk 3', 'Bulk 5', 'Bulk 6'], axis=1)
data = data.drop(['Wire 3', 'Wire 5', 'Wire 6', 'Wire 7', 'Wire 8', 'Wire 9'], axis=1)


# In[63]:


data.info()


# We calculate correlations for different columns and visualize them using a heat map.

# In[64]:


pd.set_option('display.max_columns', None)

numeric_col = data.columns.values.tolist()

corr = data.loc[:,numeric_col].corr()


# In[65]:


f, ax = plt.subplots(figsize=(35, 30))

font = {'size': 14}

plt.rc('font', **font)

cmap = sns.diverging_palette(220, 10, as_cmap=True)

ax = sns.heatmap(
    corr,         
    cmap=cmap,     
    annot=True,    
    vmax=1,       
    vmin=0,      
    center=0,      
    square=True,   
    linewidths=0, 
    xticklabels=True, yticklabels=True
)


# We see a high correlation between Bulk 9 and Wire 8.

# We are removing columns that show suspiciously high correlation.

# In[66]:


data = data.drop(['Bulk 9', 'sum_reactive_power'], axis=1)
data = data.drop(['Начало нагрева дугой', 'Конец нагрева дугой'], axis=1)


# In[67]:


data


# In[68]:


data.info()


# We have prepared the data to start predicting the temperature using various models.

# In[69]:


# We define a threshold for outliers
temperature_threshold = 1400

# We flag the outliers in the 'end_temperature' column
data['is_outlier'] = data['end_temperature'] < temperature_threshold

# We separate the outliers and non-outliers
outliers = data[data['is_outlier']]
non_outliers = data[~data['is_outlier']]

# We perform our analysis on non-outliers or outliers as needed
# For example, we can calculate the mean temperature for non-outliers:
mean_temperature = non_outliers['end_temperature'].mean()

# We can also analyze the outliers separately if needed
# For example, we can count the number of outliers:
num_outliers = outliers.shape[0]



# In[70]:


non_outliers


# ![image.png](attachment:image.png)

# In[71]:


filtered_features = non_outliers


# In[73]:


RANDOM_STATE = 120623


# In[74]:


# Adding columns 'heating_duration' and 'electrode_duration' to DataFrame filtered_features

filtered_features['heating_duration'] = data['heat_duration']
filtered_features['electrode_duration'] = data['electrode_duration']

features = data.drop('end_temperature', axis=1)
target = data['end_temperature']

Q1 = np.percentile(features, 25, axis=0)
Q3 = np.percentile(features, 75, axis=0)

IQR = Q3 - Q1

upper_bound = Q3 + (4 * IQR)

outliers_mask = (features <= upper_bound).all(axis=1)

filtered_features = features.loc[outliers_mask]
filtered_target = target.loc[outliers_mask]

max_iterations = 2329
iterations = 0

while len(filtered_features) < 2329 and iterations < max_iterations:
    upper_bound += 1
    outliers_mask = (features <= upper_bound).all(axis=1)
    filtered_features = features.loc[outliers_mask]
    filtered_target = target.loc[outliers_mask]
    iterations += 1

filtered_features = filtered_features.reset_index(drop=True)
filtered_target = filtered_target.reset_index(drop=True)

features_train, features_test, target_train, target_test = train_test_split(
    filtered_features,
    filtered_target,
    test_size=0.25,
    random_state=RANDOM_STATE
)

print("Количество оставшихся ведер:", len(filtered_features))


# In[75]:


features_train


# In[76]:


filtered_features=filtered_features.drop(['key'], axis=1)


# In[77]:


filtered_features=filtered_features.drop(['is_outlier'], axis=1)


# In[ ]:


filtered_features


# ![image.png](attachment:image.png)

# As a first option, we're going to use a random forest to see how it works.

# In[78]:


model = RandomForestRegressor() 


model.fit(features_train, target_train)

# getting important functions

importance = model.feature_importances_

importance = np.sort(importance)

# generalizing importance of the function

for i, v in enumerate(importance):
    print('Feature: {}, Score: {}'.format(i, v))
    
# the importance of the object for plotting

plt.bar([x for x in range(len(importance))], importance)
plt.show()

# Setting the threshold value of the importance of the function

threshold = 0.01

# Calculation of average importance for the function

mean_importances = np.mean(importance)

# We create a mask to identify particularly important objects

mask = mean_importances >= threshold

# Convert mask to numeric array

mask = np.array(mask)

# Convert features_train data frame to NumPy array

features_train_array = features_train.to_numpy()

# Selecting the most important objects from the dataset

filtered_features_train = features_train_array[:, mask]

# Changing the shape of the filtered_features_train array

reshaped_features_train = np.reshape(filtered_features_train, (filtered_features_train.shape[0], -1))


# In[79]:


reshaped_features_train


# ******

# In[80]:


get_ipython().run_cell_magic('time', '', '\nmodel = RandomForestRegressor() \n\nparams = [{\'criterion\':[\'mae\'],\n           \'n_estimators\':[x for x in range(100, 200, 10)], \n           \'random_state\':[RANDOM_STATE]}]\n\nclf = GridSearchCV(model, params, scoring=\'neg_mean_absolute_error\', cv=5)\n\nclf.fit(reshaped_features_train, target_train)\n\nmeans = clf.cv_results_[\'mean_test_score\']\n\nstds = clf.cv_results_[\'std_test_score\']\n\nfor mean, std, params in zip(means, stds, clf.cv_results_[\'params\']):\n    \n    print("%0.4f for %r"% ((mean*-1), params))\n    \nprint()\n\ncv_MAE_RFR = (max(means)*-1)\n\nprint()\n\nprint("Лучшие параметры")\n\nprint()\n\nprint(clf.best_params_)\n')


# The best result is obtained with 100 n_estimators, 6.84. However, we are going to continue testing with other models, such as CatBoostRegressor.

# In[81]:


get_ipython().run_cell_magic('time', '', "\nmodel = CatBoostRegressor(verbose=False, random_state=RANDOM_STATE)\n\ncv_MAE_CBR = (cross_val_score(model, \n                             reshaped_features_train, \n                             target_train, \n                             cv=5, \n                             scoring='neg_mean_absolute_error').mean() * -1)\n\nprint('MAE CatBoostRegressor =', cv_MAE_CBR)\n\nbest_params_CBR = CatBoostRegressor(verbose=False, \n                                    random_state=RANDOM_STATE).fit(features_train, \n                                        target_train).get_all_params()\n")


# CatBoost gives a slightly better result.

# In[82]:


get_ipython().run_cell_magic('time', '', '\nmodel = XGBRegressor(random_state=RANDOM_STATE)\n\n\nparams = {\n    \'n_estimators\': [100, 200, 300], \n    \'learning_rate\': [0.1, 0.01, 0.001],\n    \'max_depth\': [3, 4, 5]\n}\n\n\nclf = RandomizedSearchCV(model, params, scoring=\'neg_mean_absolute_error\', cv=5, n_iter=10)\n\nclf.fit(reshaped_features_train, target_train)\n\nprint("Наилучшие параметры для XGBoost:")\n\nprint(clf.best_params_)\n\ntarget_train_pred = clf.predict(reshaped_features_train)\n\nmae_train = mean_absolute_error(target_train, target_train_pred)\n\nprint("MAE на тренировочном наборе для XGB: {:.4f}".format(mae_train))\n')


# Scoring with XGBoost is improving.

# In[83]:


get_ipython().run_cell_magic('time', '', '\n\nmodel = GradientBoostingRegressor(random_state=RANDOM_STATE)\n\n\nparams = {\n    \'n_estimators\': [100, 200, 300],  \n    \'learning_rate\': [0.1, 0.01, 0.001],\n    \'max_depth\': [3, 4, 5]\n}\n\n\nclf = GridSearchCV(model, params, scoring=\'neg_mean_absolute_error\', cv=5)\n\nclf.fit(reshaped_features_train, target_train)\n\n\nprint("Best parameters for Gradient Boosting:")\n\nprint(clf.best_params_)\n\ntarget_train_pred = clf.predict(reshaped_features_train)\n\nmae_train = mean_absolute_error(target_train, target_train_pred)\n\nprint("MAE на тренировочном наборе для Gradient Boosting: {:.4f}".format(mae_train))\n')


# Gradient Boosting gets less as a result than the previous one, that is, the model works better.

# In[84]:


get_ipython().run_cell_magic('time', '', '\nregressor = LGBMRegressor() \n\nhyperparams = [{\'num_leaves\':[x for x in range(10,15)], \n                \'learning_rate\':[0.05, 0.07, 0.9],\n                \'random_state\':[RANDOM_STATE]}]\n\nclf = GridSearchCV(regressor, hyperparams, scoring=\'neg_mean_absolute_error\', cv=5)\n\nclf.fit(reshaped_features_train, target_train)\n\nprint("Таблица оценок по набору для разработки:")\n\nprint()\n\nmeans = clf.cv_results_[\'mean_test_score\']\n\nstds = clf.cv_results_[\'std_test_score\']\n\nfor mean, std, params in zip(means, stds, clf.cv_results_[\'params\']):\n    \n    print("%0.4f for %r"% ((mean*-1), params))\n    \nprint()\n\ncv_MAE_LGBMR = (max(means)*-1)\n\nprint()\n\nprint("Лучшие параметры:")\n\nprint()\n\nbest_params_LGBMR = clf.best_params_\n\nprint(clf.best_params_)\n  \n')


# The LGBM regressor gets the best score of 6.37.

# # Testing the best model

# We are moving on to testing the best model at the testing stage.

# In[85]:


get_ipython().run_cell_magic('time', '', '\nparams = {\n    \n    \'learning_rate\': [0.1],\n    \'max_depth\': [4],\n    \'n_estimators\': [100]\n}\n\nmodel = GradientBoostingRegressor(random_state=RANDOM_STATE)\n\nclf = GridSearchCV(model, params, scoring=\'neg_mean_absolute_error\', cv=5)\n\nclf.fit(reshaped_features_train, target_train)\n\npredictions = clf.predict(features_test)\n\nmae = mean_absolute_error(target_test, predictions)\n\nprint("MAE GradientBoostingRegressor на тестовом наборе: ", mae)         \n')


# In[87]:


final_model = XGBRegressor(random_state=RANDOM_STATE)

# Training the final model using RandomForestRegressor

final_model.fit(features_train, target_train)

# Calculation of the significance of features

feature_importances = final_model.feature_importances_
feature_importances = feature_importances[:len(filtered_features.columns)]

# Creating a DataFrame to store the significance of features

feature_importances_df = pd.DataFrame({'Признак': filtered_features.columns[:len(feature_importances)],
                                       'Значимость': feature_importances})

# Sorting Data Frame by importance in descending order

feature_importances_df = feature_importances_df.sort_values('Значимость', ascending=False)

# Plotting the significance of features

plt.figure(figsize=(10, 6))
plt.bar(feature_importances_df['Признак'], feature_importances_df['Значимость'])
plt.xlabel('Признаки')
plt.ylabel('Значимость')
plt.title('Значимость признаков')
plt.xticks(rotation=90)
plt.show()


# At the testing stage, Gradient Boosting does not meet the necessary requirements.

# In[86]:


get_ipython().run_cell_magic('time', '', '\nparams = {\n    \n    \'learning_rate\': [0.1],\n    \'max_depth\': [3],\n    \'n_estimators\': [200]\n}\n\nmodel = XGBRegressor(random_state=RANDOM_STATE)\n\nclf = GridSearchCV(model, params, scoring=\'neg_mean_absolute_error\', cv=5)\n\nclf.fit(reshaped_features_train, target_train)\n\npredictions = clf.predict(features_test)\n\nmae = mean_absolute_error(target_test, predictions)\n\nprint("MAE XGBRegressor на тестовом наборе: ", mae)         \n')


# The model that works best during the testing phase is the XGBRegressor model

# ### Conclusion
# Among the tested models, the one that seemed to fit its functions best was GradientBosstingRegressor, however, with the tested data, it did not seem to meet the requirements, so we had to test the second best model, which was XBGRegressor, than if it had been in the parameters specified by the customer.
