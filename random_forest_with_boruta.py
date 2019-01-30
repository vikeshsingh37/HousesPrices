#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.ensemble import RandomForestRegressor
from boruta import BorutaPy as bp
from sklearn.metrics import r2_score
from numpy import nanmedian
from sklearn.preprocessing import OneHotEncoder


# In[2]:


def impute_data_using_train_data(training_data, new_data, cols_to_impute):

    # Impute missing numerical columns by median
    numeric_cols = training_data.select_dtypes(include = ["float16","float32","float64","int16","int32","int64"]).columns.tolist()
    num_cols_for_imputing = list(set(numeric_cols).intersection(cols_to_impute))
    if len(num_cols_for_imputing)> 0:
        for col in num_cols_for_imputing:
            col_median = nanmedian(training_data[col])
            new_data[col].replace(np.nan,col_median, inplace = True)

    # Check categorical remaining missing values with median
    cat_cols = training_data.select_dtypes(include = ["category","object"]).columns.tolist()
    cat_cols_for_imputing = list(set(cat_cols).intersection(cols_to_impute))
    if len(cat_cols_for_imputing)> 0:
        for col in cat_cols_for_imputing:
            most_frequent = training_data[col].value_counts().axes[0].tolist()[0]
            new_data[col].replace(np.nan,most_frequent, inplace = True)

    return(new_data)

# Encode labels of categorical columns
def encode_label_categorical_cols(training_data, new_data, cols_to_encode):
    return_df = pd.DataFrame()
    for col in cols_to_encode:
        one_hot_enc = OneHotEncoder()
        one_hot_enc.fit(training_data[col].values.reshape(-1,1))
        temp = one_hot_enc.transform(new_data[col].values.reshape(-1,1)).toarray()
        dfOneHot = pd.DataFrame(temp, columns = [col+str(int(i)) for i in range(temp.shape[1])])
        return_df = pd.concat([return_df,dfOneHot], axis = 1)
        new_data = new_data.drop(columns=col, axis = 1)
        #new_data = pd.concat([new_data, dfOneHot], axis=1)
    new_data = pd.concat([new_data,return_df], axis = 1)
    return(new_data)


# In[3]:


# Load data
train_data = pd.read_csv("D:\\Kaggle\\Housing Prices\\train.csv")
test_data = pd.read_csv("D:\\Kaggle\\Housing Prices\\test.csv")


# In[4]:


train_data = train_data.sample(frac=1).reset_index(drop=True)
X = train_data.drop(columns = ["SalePrice","Id"])
y = train_data['SalePrice']


# In[5]:


# Find columns with missing values
missing_value_columns = X.isnull().sum(axis = 0)
good_columns = missing_value_columns[missing_value_columns == 0]
missing_value_columns =  missing_value_columns[missing_value_columns > 0]
miss_cols_precent = 100*missing_value_columns/X.shape[0]
print(miss_cols_precent)


# In[6]:


# Housing prices are influenced by neighborhood. So check if high missing values have a relationship with Neighborhood
cols_to_analyse = miss_cols_precent[miss_cols_precent > 10].axes[0].tolist()
cols_to_analyse.append("Neighborhood")


# In[7]:


# Get the count of missing values per neighborhood
cur_col_to_analyse = [cols_to_analyse[i] for i in [0,len(cols_to_analyse)-1] ]
X[X[cur_col_to_analyse[0]].isnull()][cur_col_to_analyse].fillna(-1).groupby("Neighborhood").count().sort_values(cur_col_to_analyse[0], ascending = False)


# In[8]:


# Get the count of missing values per neighborhood
cur_col_to_analyse = [cols_to_analyse[i] for i in [1,len(cols_to_analyse)-1] ]
X[X[cur_col_to_analyse[0]].isnull()][cur_col_to_analyse].fillna(-1).groupby("Neighborhood").count().sort_values(cur_col_to_analyse[0], ascending = False)


# In[9]:


# Get the count of missing values per neighborhood
cur_col_to_analyse = [cols_to_analyse[i] for i in [2,len(cols_to_analyse)-1] ]
X[X[cur_col_to_analyse[0]].isnull()][cur_col_to_analyse].fillna(-1).groupby("Neighborhood").count().sort_values(cur_col_to_analyse[0], ascending = False)


# In[10]:


# Get the count of missing values per neighborhood
cur_col_to_analyse = [cols_to_analyse[i] for i in [3,len(cols_to_analyse)-1] ]
X[X[cur_col_to_analyse[0]].isnull()][cur_col_to_analyse].fillna(-1).groupby("Neighborhood").count().sort_values(cur_col_to_analyse[0], ascending = False)


# In[11]:


# Get the count of missing values per neighborhood
cur_col_to_analyse = [cols_to_analyse[i] for i in [4,len(cols_to_analyse)-1] ]
X[X[cur_col_to_analyse[0]].isnull()][cur_col_to_analyse].fillna(-1).groupby("Neighborhood").count().sort_values(cur_col_to_analyse[0], ascending = False)


# In[12]:


# Get the count of missing values per neighborhood
cur_col_to_analyse = [cols_to_analyse[i] for i in [5,len(cols_to_analyse)-1] ]
X[X[cur_col_to_analyse[0]].isnull()][cur_col_to_analyse].fillna(-1).groupby("Neighborhood").count().sort_values(cur_col_to_analyse[0], ascending = False)


# In[13]:


# High missing values are spread all over the neighborhood. So, removing them from feature selection. Rest may be imputed
remaining_missing_value_columns = miss_cols_precent[miss_cols_precent < 40]
remaining_missing_value_columns = remaining_missing_value_columns.axes[0].tolist()
cols_to_keep = good_columns.axes[0].tolist() + remaining_missing_value_columns
X = X[cols_to_keep]


# In[14]:


# Check the frequencies of character column. If the frequencies are close to number of training rows or there is huge imbalance
# in categories these will be of no use in model building
cat_cols = X.select_dtypes(include = ["category","object"]).columns.tolist()
for col in cat_cols:
    print(X[col].fillna(-1).value_counts())


# In[15]:


# Columns to remove with high imbalance in categories
high_imb_cats = ['RoofMatl', 'Condition2', 'Utilities', 'Street']
X.drop(columns = high_imb_cats, axis = 1, inplace = True)


# In[16]:


cat_cols = [i for i in cat_cols if i not in high_imb_cats]


# In[17]:


# Impute missing numerical columns by median and categorical by most frequent
X_train = X
X = impute_data_using_train_data(X_train,X,remaining_missing_value_columns)


# In[18]:


# Check if still there are missing values
nan_check = X[remaining_missing_value_columns].isnull().values.any()
print(nan_check)

if(nan_check):
    print("There are still NA values")
    #sys.exit()


# In[ ]:


# Convert string columns to one hot encoded vector
X = encode_label_categorical_cols(training_data = X_train, new_data = X, cols_to_encode=cat_cols)


# In[ ]:


# Feature selection by Boruta
rf_model = RandomForestRegressor (n_jobs= 2,oob_score= True)
feat_selector = bp(rf_model,n_estimators = 'auto', verbose= 1,max_iter= 1000)
feat_selector.fit(X.values, y.values)


# In[ ]:


# Evaluate Random Forest Model
selected_features = X.columns[feat_selector.support_].tolist()
print(selected_features)
rf_model = RandomForestRegressor(n_estimators=100, n_jobs= 2,oob_score= True, max_features= 5)
model = rf_model.fit(X[selected_features], y.values)
model.oob_score_


# In[ ]:


# Prepare test data for predictions
columnsTitles = train_data.columns.tolist()
columnsTitles.remove("SalePrice")
test_data = test_data[columnsTitles]
test_data = test_data.reindex(columns=columnsTitles)
X_test = test_data
# Impute data from training data
X_test = impute_data_using_train_data(X_train,X_test,columnsTitles)

# Encode categorical data
X_test = encode_label_categorical_cols(training_data = X_train, new_data = X_test, cols_to_encode=cat_cols)
X_test = X_test[selected_features]
X_test.shape


# In[ ]:


# Predict 
y_predict = model.predict(X_test)


# In[ ]:


test_data["SalePrice"] = y_predict.tolist()
final_data = test_data[["Id","SalePrice"]]
final_data.to_csv("D:\\Kaggle\\Housing Prices\\predictions.csv", index=False)

