#!/usr/bin/env python
# coding: utf-8

# # Mercari Price Suggest
# This Machine Learning Project allows an online seller to enter their product and its information. It then predicts a price suggestion for what the seller should sell that product.
# 
# 
# The main steps of this project:
# 
# 1. Understand the problem definition
# 
# 2. Get the data
# 
# 3. Discover and visualize the data to gain insights
# 
# 4. Prepare the data for Machine Learning Algorithms
# 
# 5. Select a model and train it
# 
# 6. Fine-tune and provide a solution
# 
# 
# 
# ## Problem Definition
# The training data contains the ids, item_name, category_name, price, shipping, item_condition_id, and item_description. Java is not an optimal language for the project since the dataset is considerably large with more than 1 million instances. A more advanced language such as Python serves as a better alternative for handling larger datasets. Python also has in-built functions, like pandas, for handling large datasets.
# 
# 
# ## Getting the Data
# The train, test, and sample datasets (which were in both csv and tsv formats) were taken from Kaggle. Pandas Library, which stands for Python Data Analysis Library was used to examine the multidimensional structured datasets. The Python Data Analysis Library inputs the csv or tsv file to create the Data Frame, which is a Python object that stores rows and columns.
#  
# ## Discover and Visualizing the Data to Gain Insights
# The software Tableu and the built-in Pyplot feature to visualize the data and find the relationship between categories and the other factors such as price, shipping, and item_condition_id.
# 
# ## Preparing the Data for Machine Learning Algorithms
# The dataset was reduced to only having the category_names starting with 'Electronics' since the dataset was relatively large to begin with. The training data was used to find patterns in the fields category_name, price, shipping, item_condition_id.
# 
# ## Selecting the Model
# The Linear Regression model was used to find trends/correlations categorical fields such as category_name, shipping, item_condition_id and the price. The command used to import Linear Regression was: from sklearn.linear_model import LinearRegression.
# 
# Feature Extraction was also used for taking the raw categorical data, and reducing it to more manageable groups. This method was mainly used in the category_name and brand_name fields thorugh one hot encoding. One hot encoding gives a numerical characteristic to the categorical variables through values 1 and 0.
# 
# ## Load Data Using Python Pandas Library
# ## Use sklearn library to find Linear Regression

# In[1]:


import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import KFold


# ## Reads Training Set

# In[2]:


df = pd.read_csv('train.csv',encoding="ISO-8859-1")


# Print DataFrame to show sample entries

# ## Data Structures: Data Frame
# 

# In[3]:


df


# Observe that fields like category_name and brand_name are not present in all the instances.

# In[4]:


df.info()


# In[5]:


df["category_name"].value_counts()


# In[6]:


df["brand_name"].value_counts()


# In[7]:


df.describe()


# In[8]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
df.hist(bins=20, figsize=(20,15))
plt.show()


# Extract entries of category_name that start with 'Electronics' because this reduces the amount of data sets from ~1000000 to ~90000, allowing the computer to run the training data more efficiently.

# In[9]:


df_Electronics = df[df['category_name'].str.startswith('Electronics', na=False)]


# Printing electronics dataframe.

# In[10]:


df_Electronics


# Extract one hot vector encoding to differentiate between each category (category_name and brand_name), using Pandas.

# In[11]:


dummies1 = pd.get_dummies(df_Electronics.category_name)
dummies2 = pd.get_dummies(df_Electronics.brand_name)


# Printing both category_name and brand_name dummies

# In[12]:


dummies1
dummies2


# Merging both df_Electronics and dummies

# In[13]:


merged = pd.concat([df_Electronics, dummies1, dummies2], axis = 'columns')


# In[14]:


merged


# Creating 'final' dataset in which train_id, name, brand_name, item_description, category_name, and Electronics/Cameras & Photography/Binoculars & Telescopes are dropped. Dropping train_id, name, brand_name, item_description, category_name allows the program to train the data to find linear regressions between ('item_condition_id', 'shipping', 'brand_name', and 'category_name') and 'price'.

# In[15]:


final = merged.drop(['train_id', 'name', 'brand_name', 'item_description', 'category_name', 'Electronics/Cameras & Photography/Binoculars & Telescopes'], axis = 'columns')
final


# Developing linear regression model to find correlations between category_name, brand_name, and price.

# In[16]:


from sklearn.linear_model import LinearRegression
model = RandomForestRegressor(n_estimators = 100)


# In[17]:


X=final.drop('price',axis='columns')
X


# In[18]:


Y=final.price
Y


# Finding linear regression between X and Y

# In[22]:


model.fit(X,Y)


# In[19]:


#scores = []
#kfold = KFold(n_splits=3, shuffle=True, random_state=42)
#for i, (train, test) in enumerate(kfold.split(X, Y)):
    #model.fit(X.iloc[train,:], Y.iloc[train,:])
    #score = model.score(X.iloc[test,:], Y.iloc[test,:])
    #scores.append(score)
    #print(scores)



score = model.score(X, Y)
score


# In[20]:


predictions = model.predict(X)
plt.scatter(Y, predictions)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.xlim([0,250])


# In[21]:


final.hist(column = 'price', bins = 1000)
plt.xlim([0,50])


# # Using Test Data to Find Model Accuracy

# In[ ]:


df_test = pd.read_csv('test.csv',encoding="ISO-8859-1")


# Extract entries of category_name that start with 'Electronics'

# In[ ]:


df_test_Electronics = df_test[df_test['category_name'].str.startswith('Electronics', na=False)]


# Printing df_test_Electronics

# In[ ]:


df_test_Electronics


# Extract one hot vector encoding to differentiate between each category (category_name and brand_name) using 'dummies_test' variables for the test data.

# In[ ]:


dummies_test1 = pd.get_dummies(df_test_Electronics.category_name)
dummies_test1


# In[ ]:


dummies_test2 = pd.get_dummies(df_test_Electronics.brand_name)
dummies_test2


# Merging both df_test_Electronics with dummies_test1 and dummies_test2

# In[ ]:


merged_test = pd.concat([df_test_Electronics, dummies_test1, dummies_test2], axis = 'columns')


# In[ ]:


merged_test


# Making final test data set by dropping 'test_id', 'name', 'item_description', and 'category_name' to not have Strings when trying to find linear regressions between variables.

# In[ ]:


final_test = merged_test.drop(['test_id', 'name', 'brand_name', 'item_description', 'category_name', 'Electronics/Cameras & Photography/Binoculars & Telescopes'], axis = 'columns')
final_test


# In[ ]:


X_test = final_test


# In[ ]:


missing_cols = set(X.columns) - set(X_test.columns)


# In[ ]:


missing_cols


# Fill the missing columns in the test matrix and assigned missing columns to be 0.

# In[ ]:


for c in missing_cols:
    X_test[c] = 0


# Make X_test column order same as X column order. Model is trained on X column order.

# In[ ]:


X_test = X_test[X.columns]


# Printing X_test

# In[ ]:


X_test


# In[ ]:


Y_test = model.predict(X_test)


# Printing Y_test

# In[ ]:


Y_test


# In[ ]:


df_Y_test= pd.DataFrame(data = Y_test, columns = ['price'])


# In[ ]:


df_Y_test


# ## Predicting the Price of the Sample Data

# In[ ]:


cols = [0]
df_submissions = merged_test[merged_test.columns[cols]]


# In[ ]:


j=0
for i, row in df_submissions.iterrows():
    df_submissions.at[i, 'price'] = Y_test[j]
    j = j+1


# In[ ]:


df_submissions.count


# In[ ]:


export_csv = df_submissions.to_csv('C:/WPy64-3760/notebooks/Kanishk_Mercari/Mercari_Price_Output.csv', index = False)


# In[ ]:




