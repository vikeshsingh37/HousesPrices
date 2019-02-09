#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.chdir("D:\Github\HousesPrices")
import data_preprocessing as dp
import feature_selection as fs
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor,ExtraTreesRegressor
import numpy as np
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import cross_validate
import math
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_log_error
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.ensemble import ExtraTreesRegressor


# In[2]:


def cv_error_for_log_transformed_data(X,y,test_model,cv_folds =5):
    rkf = RepeatedKFold(n_splits=cv_folds, n_repeats=1, random_state=2652124)
    err = []
    for train_index, test_index in rkf.split(X):
        X_train, X_testing = X.iloc[train_index,:], X.iloc[test_index,:]
        y_train, y_testing = y[train_index], y[test_index]
        test_model.fit(X_train,y_train)
        pred = test_model.predict(X_testing)
        err.append(np.sqrt(mean_squared_log_error(np.expm1(y_testing), np.expm1(pred))))
    return(np.mean(err))


# In[3]:


dp_object = dp.data_preprocessing("D:/Kaggle/Housing Prices/train.csv","D:/Kaggle/Housing Prices/test.csv")
X,X_test,y = dp_object.get_processed_data()


# In[4]:


fs_object = fs.feature_selection(X,y)
selected_features = fs_object.boruta_fs(n_estimators = 'auto', max_iter = 1000, n_jobs = 7)


# In[5]:


rf_model = RandomForestRegressor(n_estimators=1000, n_jobs= 7,oob_score= True, random_state=42,max_features=6)
rf_model.fit(X[selected_features], y.values)
print(rf_model.oob_score_)

cv_results_rf = cross_validate(rf_model, X[selected_features], y.values, cv=10,scoring = 'neg_mean_squared_log_error',
                         return_train_score=False)

print(np.mean([math.sqrt(-x) for x in cv_results_rf['test_score'].tolist()]))

print(cv_error_for_log_transformed_data(X[selected_features],y,rf_model,cv_folds =10))


# In[6]:


y_predict = rf_model.predict(X_test[selected_features])
y_predict_rf = y_predict.tolist()
y_predict_rf = np.expm1(y_predict_rf)
X_test["SalePrice"] = y_predict_rf
final_data = X_test[["Id","SalePrice"]]
final_data.to_csv("D:\\Kaggle\\Housing Prices\\predictions_boruta_rf.csv", index=False)


# In[7]:


gbm_model = GradientBoostingRegressor(n_estimators=3000, max_depth = 4, learning_rate=0.01, random_state=42,max_features = 3,
                                     loss='huber')
gbm_model.fit(X[selected_features], y.values)
print(gbm_model.score(X[selected_features],y.values))

from sklearn.model_selection import cross_validate
cv_results_gbm = cross_validate(gbm_model, X[selected_features], y.values, cv=10,scoring = 'neg_mean_squared_log_error',
                         return_train_score=False)

print(np.mean([math.sqrt(-x) for x in cv_results_gbm['test_score'].tolist()]))
print(cv_error_for_log_transformed_data(X[selected_features],y,gbm_model,cv_folds =10))


# In[8]:


y_predict = gbm_model.predict(X_test[selected_features])
y_predict_gbm = np.expm1(y_predict.tolist())
X_test["SalePrice"] = np.expm1(y_predict.tolist())
final_data = X_test[["Id","SalePrice"]]
final_data.to_csv("D:\\Kaggle\\Housing Prices\\predictions_boruta_gbm.csv", index=False)


# In[9]:


xgb_model = xgb.XGBRegressor(objective="reg:linear", n_estimators=3000, max_depth = 4, learning_rate=0.01, random_state=42,
                             reg_alpha=0, reg_lambda=0,n_jobs=4,colsample_bytree = 0.3)

xgb_model.fit(X[selected_features], y.values)

cv_results_xgb = cross_validate(xgb_model, X[selected_features], y.values, cv=10,scoring = 'neg_mean_squared_log_error',
                         return_train_score=False)
print(np.mean([math.sqrt(-x) for x in cv_results_xgb['test_score'].tolist()]))
print(cv_error_for_log_transformed_data(X[selected_features],y,xgb_model,cv_folds =10))


# In[10]:


y_predict = xgb_model.predict(X_test[selected_features])
y_predict_xgb = np.expm1(y_predict.tolist())
X_test["SalePrice"] = np.expm1(y_predict.tolist())
final_data = X_test[["Id","SalePrice"]]
final_data.to_csv("D:\\Kaggle\\Housing Prices\\predictions_boruta_xgb.csv", index=False)


# In[11]:


et_model = ExtraTreesRegressor(n_estimators=1000, n_jobs= 7, max_features= 6, random_state=42)
et_model.fit(X[selected_features], y.values)
#print(et_model.oob_score_)
cv_results_et = cross_validate(et_model, X[selected_features], y.values, cv=10,scoring = 'neg_mean_squared_log_error',
                         return_train_score=False)

print(np.mean([math.sqrt(-x) for x in cv_results_et['test_score'].tolist()]))
print(cv_error_for_log_transformed_data(X[selected_features],y,et_model,cv_folds =10))


# In[12]:


y_predict = et_model.predict(X_test[selected_features])
y_predict_et = np.expm1(y_predict.tolist())
X_test["SalePrice"] = np.expm1(y_predict.tolist())
final_data = X_test[["Id","SalePrice"]]
final_data.to_csv("D:\\Kaggle\\Housing Prices\\predictions_boruta_et.csv", index=False)


# In[13]:


lgb_model = lgb.LGBMRegressor(n_estimators=3000, learning_rate=0.01, random_state=42,boosting_type = 'gbdt',max_depth= 4,
                              colsample_bytree = 0.4,reg_alpha=0, reg_lambda=0, n_jobs =7)
lgb_model.fit(X[selected_features], y.values)
cv_results_et = cross_validate(lgb_model, X[selected_features], y.values, cv=10,scoring = 'neg_mean_squared_log_error',
                         return_train_score=False)

print(np.mean([math.sqrt(-x) for x in cv_results_et['test_score'].tolist()]))
print(cv_error_for_log_transformed_data(X[selected_features],y,lgb_model,cv_folds =10))


# In[14]:


y_predict = lgb_model.predict(X_test[selected_features])
y_predict_lgb = np.expm1(y_predict.tolist())
X_test["SalePrice"] = np.expm1(y_predict.tolist())
final_data = X_test[["Id","SalePrice"]]
final_data.to_csv("D:\\Kaggle\\Housing Prices\\predictions_boruta_lgb.csv", index=False)


# In[15]:


avg_pred_from_all_models = pd.DataFrame({'ID': X_test["Id"]})
#avg_pred_from_all_models["rf"] = y_predict_rf
avg_pred_from_all_models["gbm"] = y_predict_gbm
avg_pred_from_all_models["xgb"] = y_predict_xgb
#avg_pred_from_all_models["et"] = y_predict_et
avg_pred_from_all_models["lgb"] = y_predict_lgb
avg_pred_from_all_models['SalePrice'] = avg_pred_from_all_models.drop(columns = "ID").mean(axis=1)
avg_pred_from_all_models
avg_pred_from_all_models[["ID","SalePrice"]].to_csv("D:\\Kaggle\\Housing Prices\\predictions_boruta_avg.csv", index=False)

