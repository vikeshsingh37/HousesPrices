#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.chdir("D:\Github\HousesPrices")
from stack_model import stack_model
from hyper_parameter_tuning import tune_ga
import data_preprocessing as dp
import feature_selection as fs
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor,ExtraTreesRegressor
import numpy as np
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import cross_validate
import math
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_log_error,mean_squared_error
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.ensemble import ExtraTreesRegressor
import tpot as tp
from sklearn.model_selection import cross_validate
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split, KFold


# In[2]:


dp_object = dp.data_preprocessing("D:/Kaggle/Housing Prices/train.csv","D:/Kaggle/Housing Prices/test.csv")
X,X_test,y = dp_object.get_processed_data()


# In[3]:


fs_object = fs.feature_selection(X,y)
selected_features = fs_object.boruta_fs(n_estimators = 'auto', max_iter = 1000, n_jobs = 7)


# In[4]:


selected_features


# In[5]:


rf_model = RandomForestRegressor(n_estimators=1000, n_jobs= 7,oob_score= True, random_state=42,max_features=6)
rf_model.fit(X[selected_features], y.values)
print(rf_model.oob_score_)

cv_results_rf = cross_validate(rf_model, X[selected_features], y.values, cv=10,scoring = 'neg_mean_squared_error',
                         return_train_score=False)

print(np.mean(np.sqrt(-cv_results_rf['test_score'])))
print(np.std(np.sqrt(-cv_results_rf['test_score'])))


# In[6]:


y_predict = rf_model.predict(X_test[selected_features])
y_predict_rf = y_predict.tolist()
y_predict_rf = np.expm1(y_predict_rf)
X_test["SalePrice"] = y_predict_rf
final_data = X_test[["Id","SalePrice"]]
final_data.to_csv("D:\\Kaggle\\Housing Prices\\predictions_boruta_rf.csv", index=False)


# In[7]:


gbm_model = GradientBoostingRegressor(n_estimators=1500, max_depth = 4, learning_rate=0.01, random_state=42,max_features = 3,
                                     loss='huber')
gbm_model.fit(X[selected_features], y.values)
print(gbm_model.score(X[selected_features],y.values))

from sklearn.model_selection import cross_validate
cv_results_gbm = cross_validate(gbm_model, X[selected_features], y.values, cv=10,scoring = 'neg_mean_squared_error',
                         return_train_score=False)

print(np.mean(np.sqrt(-cv_results_gbm['test_score'])))
print(np.std(np.sqrt(-cv_results_gbm['test_score'])))


# In[8]:


y_predict = gbm_model.predict(X_test[selected_features])
y_predict_gbm = np.expm1(y_predict.tolist())
X_test["SalePrice"] = y_predict_gbm
final_data = X_test[["Id","SalePrice"]]
final_data.to_csv("D:\\Kaggle\\Housing Prices\\predictions_boruta_gbm.csv", index=False)


# In[9]:


xgb_model = xgb.XGBRegressor(objective="reg:linear", n_estimators=1200, max_depth = 4, learning_rate=0.01, random_state=42,
                             reg_alpha=0, reg_lambda=0,n_jobs=4,colsample_bytree = 0.3)

xgb_model.fit(X[selected_features], y.values)

cv_results_xgb = cross_validate(xgb_model, X[selected_features], y.values, cv=10,scoring = 'neg_mean_squared_error',
                         return_train_score=False)
print(np.mean(np.sqrt(-cv_results_xgb['test_score'])))
print(np.std(np.sqrt(-cv_results_xgb['test_score'])))


# In[10]:


y_predict = xgb_model.predict(X_test[selected_features])
y_predict_xgb = np.expm1(y_predict.tolist())
X_test["SalePrice"] =y_predict_xgb 
final_data = X_test[["Id","SalePrice"]]
final_data.to_csv("D:\\Kaggle\\Housing Prices\\predictions_boruta_xgb.csv", index=False)


# In[11]:


et_model = ExtraTreesRegressor(n_estimators=1000, n_jobs= 7, max_features= 6, random_state=42)
et_model.fit(X[selected_features], y.values)
#print(et_model.oob_score_)
cv_results_et = cross_validate(et_model, X[selected_features], y.values, cv=10,scoring = 'neg_mean_squared_error',
                         return_train_score=False)

print(np.mean(np.sqrt(-cv_results_et['test_score'])))
print(np.std(np.sqrt(-cv_results_et['test_score'])))


# In[12]:


y_predict = et_model.predict(X_test[selected_features])
y_predict_et = np.expm1(y_predict.tolist())
X_test["SalePrice"] =y_predict_et
final_data = X_test[["Id","SalePrice"]]
final_data.to_csv("D:\\Kaggle\\Housing Prices\\predictions_boruta_et.csv", index=False)


# In[13]:


lgb_model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.01, random_state=42,boosting_type = 'gbdt',max_depth= 4,
                              colsample_bytree = 0.6,reg_alpha=0.01, reg_lambda=0, n_jobs =7)
lgb_model.fit(X[selected_features], y.values)
cv_results_lgb = cross_validate(lgb_model, X[selected_features], y.values, cv=10,scoring = 'neg_mean_squared_error',
                         return_train_score=False)

print(np.mean(np.sqrt(-cv_results_lgb['test_score'])))
print(np.std(np.sqrt(-cv_results_lgb['test_score'])))


# In[14]:


y_predict = lgb_model.predict(X_test[selected_features])
y_predict_lgb = np.expm1(y_predict.tolist())
X_test["SalePrice"] = y_predict_lgb
final_data = X_test[["Id","SalePrice"]]
final_data.to_csv("D:\\Kaggle\\Housing Prices\\predictions_boruta_lgb.csv", index=False)


# In[15]:


en_model = ElasticNet(alpha=0.0005, l1_ratio=0.01, max_iter=7000)
en_model.fit(X[selected_features], y.values)
cv_results_en = cross_validate(en_model, X[selected_features], y.values, cv=10,scoring = 'neg_mean_squared_error',
                         return_train_score=False)

print(np.mean(np.sqrt(-cv_results_en['test_score'])))
print(np.std(np.sqrt(-cv_results_en['test_score'])))


# In[16]:


y_predict = en_model.predict(X_test[selected_features])
y_predict_en = np.expm1(y_predict.tolist())
X_test["SalePrice"] = y_predict_en
final_data = X_test[["Id","SalePrice"]]
final_data.to_csv("D:\\Kaggle\\Housing Prices\\predictions_boruta_en.csv", index=False)


# In[17]:


base_models = [rf_model,gbm_model,en_model]
meta_model = RandomForestRegressor(n_estimators =100)
kf = KFold(n_splits= 5, shuffle=False, random_state=42)
cv_score = []
for train_index, test_index in kf.split(X):
    X_tr, X_te = X[selected_features].iloc[train_index], X[selected_features].iloc[test_index]
    y_tr, y_te = y[train_index], y[test_index]
    stk_model = stack_model(base_models,meta_model,X_tr,y_tr.values)
    stk_model.fit(cvFolds = 5, random_state= 42)
    cv_score.append(mean_squared_error(y_te.tolist(),stk_model.predict(X_te)))


# In[18]:


print(np.mean(np.sqrt(cv_score)))
print(np.std(np.sqrt(cv_score)))


# In[21]:


meta_model = RandomForestRegressor(n_estimators =100)
stk_model = stack_model(base_models,meta_model,X[selected_features],y.values)
stk_model.fit(cvFolds = 5, random_state= 42)
y_predict = stk_model.predict(X_test[selected_features])
y_predict_st = np.expm1(y_predict)
X_test["SalePrice"] = y_predict_st
final_data = X_test[["Id","SalePrice"]]
final_data.to_csv("D:\\Kaggle\\Housing Prices\\predictions_boruta_stk.csv", index=False)

