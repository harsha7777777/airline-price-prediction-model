#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from termcolor import colored
import warnings

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV

print(colored('\nAll libraries imported succesfully', 'green'))


# In[2]:


pd.options.mode.copy_on_write = True # Allow re-write on variable
sns.set_style('darkgrid') # Seaborn style
warnings.filterwarnings('ignore') # Ignore warnings
pd.set_option('display.max_columns', None) # Setting this option will print all collumns of a dataframe
pd.set_option('display.max_colwidth', None) # Setting this option will print all of the data in a feature


# In[3]:


data = pd.read_csv('Clean_Dataset.csv')
data.head()


# In[4]:


data.info()


# In[5]:


data.isna().sum()


# In[6]:


data.drop(columns='Unnamed: 0', inplace=True)


# In[7]:


data.rename(columns={'class': 'flight_class'}, inplace=True)


# In[8]:


for index, value in enumerate(data.columns) :
    print(index, ":", value)


# In[9]:


data.airline.value_counts()


# In[10]:


fig, ax = plt.subplots(figsize=(8, 5))
fig.suptitle('Air Line', fontsize=20, fontweight='bold')

# Pie chart
labels = data.airline.value_counts().index.tolist()
explode = (0, 0, 0, 0, 0, 0.5)
ax.pie(data.airline.value_counts(), 
       autopct='%.f%%', 
       labels=labels, 
       shadow=True, 
       pctdistance=1.15, 
       labeldistance=0.6, 
       explode=explode)
ax.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=5)

# Display the plot
plt.tight_layout()
plt.show()



# In[11]:


data.drop(columns='flight', inplace=True)


# In[12]:


data.source_city.value_counts()


# In[13]:


fig, ax = plt.subplots(figsize=(8, 5))
fig.suptitle('Source City', fontsize=20, fontweight='bold')

# Pie chart
labels = data.source_city.value_counts().index.tolist()
explode = (0, 0, 0, 0, 0, 0.3)
ax.pie(data.source_city.value_counts(), 
       autopct='%.f%%', 
       labels=labels, 
       shadow=True, 
       pctdistance=1.2, 
       labeldistance=0.4, 
       explode=explode)
ax.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=5)

# Display the plot
plt.tight_layout()
plt.show()



# In[14]:


data.departure_time.value_counts()


# In[15]:


import matplotlib.pyplot as plt

# Create a single subplot for the pie chart
fig, ax = plt.subplots(figsize=(7, 5))
fig.suptitle('Departure Time', fontsize=20, fontweight='bold')

# Pie chart
labels = data.departure_time.value_counts().index.tolist()
explode = (0, 0, 0, 0, 0, 0.3)
ax.pie(data.departure_time.value_counts(), 
       autopct='%.f%%', 
       labels=labels, 
       shadow=True, 
       pctdistance=1.2, 
       labeldistance=0.4, 
       explode=explode)
ax.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=5)

# Display the plot
plt.tight_layout()
plt.show()


# In[16]:


data.stops.value_counts()


# In[17]:


fig, ax = plt.subplots(figsize=(7,5))
fig.suptitle('Stops', fontsize=15, fontweight='bold')

labels = data.stops.value_counts().index.tolist()
explode = (0, 0, 0.3)
ax.pie(data.stops.value_counts(), 
       autopct='%.f%%', 
       labels=labels, 
       shadow=True, 
       pctdistance=1.2, 
       labeldistance=0.4, 
       explode=explode)
ax.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=5)

plt.tight_layout()
plt.show()


# In[18]:


data.arrival_time.value_counts()


# In[19]:


fig, ax = plt.subplots(figsize=(8, 6))
fig.suptitle('Arrival Time', fontsize=20, fontweight='bold')

# Pie chart
labels = data.arrival_time.value_counts().index.tolist()
explode = (0, 0, 0, 0, 0, 0.3)
ax.pie(data.arrival_time.value_counts(), 
       autopct='%.f%%', 
       labels=labels, 
       shadow=True, 
       pctdistance=1.2, 
       labeldistance=0.4, 
       explode=explode)
ax.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=5)

# Display the plot
plt.tight_layout()
plt.show()


# In[20]:


data.destination_city.value_counts()


# In[21]:


fig, ax = plt.subplots(figsize=(7, 5))
fig.suptitle('Destination City', fontsize=20, fontweight='bold')

# Pie chart
labels = data.destination_city.value_counts().index.tolist()
explode = (0, 0, 0, 0, 0, 0.3)
ax.pie(data.destination_city.value_counts(), 
       autopct='%.f%%', 
       labels=labels, 
       shadow=True, 
       pctdistance=1.2, 
       labeldistance=0.4, 
       explode=explode)
ax.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=5)

# Display the plot
plt.tight_layout()
plt.show()


# In[22]:


data.flight_class.value_counts()


# In[23]:


fig, ax = plt.subplots(figsize=(7, 5))
fig.suptitle('Flight Class', fontsize=20, fontweight='bold')
labels = data.flight_class.value_counts().index.tolist()
ax.pie(data.flight_class.value_counts(), 
       autopct='%.f%%', 
       labels=labels, 
       shadow=True, 
       pctdistance=1.2, 
       labeldistance=0.4)
ax.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=5)

# Display the plot
plt.tight_layout()
plt.show()


# In[24]:


len(data.duration.value_counts())


# In[25]:


fig, ax = plt.subplots(1, 1, figsize=(15, 5))
plt.tight_layout()

# Left ax (Line plot)
sns.lineplot(x='duration', y='price', data=data, ax=ax, hue='flight_class').set_xticks(np.arange(0, 50, 5))

# Display the plot
plt.show()


# In[26]:


len(data.days_left.value_counts())


# In[27]:


fig, ax = plt.subplots(1, 1, figsize=(15, 5))
plt.tight_layout()

# Left ax (Line plot)
sns.lineplot(x='days_left', y='price', data=data, ax=ax).set_xticks(np.arange(0, 50, 5))

# Display the plot
plt.show()


# In[28]:


# col : airline
data.airline = data.airline.replace(
    {
        'Vistara' : 1,
        'Air_India' : 2,
        'Indigo' : 3,
        'GO_FIRST' : 4,
        'AirAsia': 5,
        'SpiceJet' : 6  
    }
)


# In[29]:


# col : source_city
data.source_city = data.source_city.replace(
    {
        'Delhi' : 1,
        'Mumbai' : 2,
        'Bangalore' : 3,
        'Kolkata' : 4,
        'Hyderabad'  : 5,
        'Chennai' : 6
    }
)


# In[30]:


# col : departure_time
data.departure_time = data.departure_time.replace(
    {
        'Morning' : 1,
        'Early_Morning' : 2, 
        'Evening' : 3,
        'Night' : 4,
        'Afternoon' : 5, 
        'Late_Night' : 6
    }
)


# In[31]:


# col : stops
data.stops = data.stops.replace(
    {
        'one' : 1,
        'zero' : 2,
        'two_or_more' : 3
    }
)


# In[32]:


# col : arrival_time
data.arrival_time = data.arrival_time.replace(
    {
        'Night' : 1,
        'Evening' : 2,
        'Morning' : 3,
        'Afternoon' : 4,
        'Early_Morning' : 5,
        'Late_Night' : 6
    }
)


# In[33]:


# col : destination_city
data.destination_city = data.destination_city.replace(
    {
        'Mumbai' : 1,
        'Delhi' : 2,
        'Bangalore' : 3,
        'Kolkata' : 4,
        'Hyderabad' : 5,
        'Chennai' : 6
    }
)


# In[34]:


# col : flight_class
data.flight_class = data.flight_class.replace(
    {
        'Economy' : 1,
        'Business' :2
    }
)


# In[35]:


data.describe()


# In[36]:


corr = data.corr()
plt.figure(figsize=(10, 5))
sns.heatmap(corr, annot=True, fmt='.2f', linewidths=0.5, linecolor='white', mask=np.triu(corr), cmap='Blues')
plt.show()


# In[37]:


X_temp = data.drop(columns='price')
y = data.price


# In[38]:


scaler = MinMaxScaler().fit_transform(X_temp)
X = pd.DataFrame(scaler, columns=X_temp.columns)
main_X = X.copy()


# In[39]:


# Create a loop to find best test_size
test_list = []
mse_list = []
r2score_list = []
best_r2=0
best_mse=0
best_test=0

for tester in range(6, 19) :
    tester = round(0.025 * tester, 2)
    test_list.append(tester)
    #
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=tester, random_state=0)
    #
    lr = LinearRegression().fit(X, y)
    y_pred_lr = lr.predict(X_test)
    r2score = metrics.r2_score(y_test, y_pred_lr)
    r2score_list.append(r2score)
    mse = metrics.mean_squared_error(y_test, y_pred_lr)
    mse_list.append(mse)
    #
    if r2score>best_r2 :
        best_r2 = r2score
        best_mse = mse
        best_test = tester
print(colored('Best test_size : {}'.format(best_test), 'blue'))
print(colored('Best R2Score : {}'.format(best_r2), 'blue'))
print(colored('Best Mean Squared Error : {}'.format(best_mse), 'blue'))

# Plot
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(test_list, r2score_list, c='blue', label='R2Score')
ax[0].set_title("R2Score")
ax[0].legend()

ax[1].plot(test_list, mse_list, c='red', label='Mean Squared Error')
ax[1].set_title("Mean Squared Error")
ax[1].legend()
plt.show()


# In[40]:


lr_r2 = best_r2
print(colored('Liear Legresion R2Score = {}'.format(round(lr_r2, 3)), 'green'))


# In[41]:


X_train, X_test, y_train, y_test = train_test_split(main_X, y, test_size=tester, random_state=0)

rf = RandomForestRegressor(n_estimators=500, max_features=8, n_jobs=-1).fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
rf_r2 = metrics.r2_score(y_test, y_pred_rf)
print(colored('RandomForestRegressor R2Score = {}'.format(round(rf_r2, 3)), 'green'))


# In[42]:


result = pd.DataFrame({
    'Algorithms': ['LinearRegression', 'RandomForestRegressor'],
    'R2Scores': [lr_r2, rf_r2]
})

# Plot the R2 Scores
plt.figure(figsize=(8, 4))
ax = sns.barplot(x='Algorithms', y='R2Scores', data=result, palette='Set1')
for container in ax.containers:
    ax.bar_label(container)
plt.title('R2 Scores for Linear Regression and Random Forest')
plt.show()

# Print R2 Scores
print(f'Linear Regression R2 Score: {lr_r2}')
print(f'Random Forest R2 Score: {rf_r2}')


# In[ ]:




