#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.chdir("D:/Kaggle/Housing Prices/")
import data_preprocessing as dp
import feature_selection as fs
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor,ExtraTreesRegressor
import numpy as np
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import cross_validate


# In[ ]:


dp_object = dp.data_preprocessing("D:/Kaggle/Housing Prices/train.csv","D:/Kaggle/Housing Prices/test.csv")
X,X_test,y = dp_object.get_processed_data()


# In[ ]:


fs_object = fs.feature_selection(X,y)
selected_features = fs_object.boruta_fs(n_estimators = 'auto', max_iter = 1000, n_jobs = 3)


# In[ ]:


rf_model = RandomForestRegressor(n_estimators=1000, n_jobs= 2,oob_score= True, max_features= 5,random_state=42)
rf_model.fit(X[selected_features], y.values)
rf_model.oob_score_

cv_results_rf = cross_validate(rf_model, X[selected_features], y.values, cv=10,scoring = 'neg_mean_squared_log_error',
                         return_train_score=False)

-np.mean(cv_results_rf['test_score'].tolist())


# In[ ]:


y_predict = rf_model.predict(X_test[selected_features])
y_predict_rf = y_predict.tolist()
X_test["SalePrice"] = y_predict.tolist()
final_data = X_test[["Id","SalePrice"]]
final_data.to_csv("D:\\Kaggle\\Housing Prices\\predictions_boruta_rf.csv", index=False)


# In[ ]:


gbm_model = GradientBoostingRegressor(n_estimators=50, max_depth = 4, learning_rate=0.11, random_state=42)
gbm_model.fit(X[selected_features], y.values)
print(gbm_model.score(X[selected_features],y.values))

from sklearn.model_selection import cross_validate
cv_results_gbm = cross_validate(gbm_model, X[selected_features], y.values, cv=10,scoring = 'neg_mean_squared_log_error',
                         return_train_score=False)

-np.mean(cv_results_gbm['test_score'].tolist())


# In[ ]:


y_predict = gbm_model.predict(X_test[selected_features])
y_predict_gbm = y_predict.tolist()
X_test["SalePrice"] = y_predict.tolist()
final_data = X_test[["Id","SalePrice"]]
final_data.to_csv("D:\\Kaggle\\Housing Prices\\predictions_boruta_gbm.csv", index=False)


# In[ ]:


import xgboost as xgb
from sklearn.metrics import r2_score
xgb_model = xgb.XGBRegressor(objective="reg:linear", random_state=42,max_depth=4, learning_rate=0.11, n_estimators=50,
                            reg_alpha=0, reg_lambda=0)

xgb_model.fit(X[selected_features], y.values)

cv_results_xgb = cross_validate(xgb_model, X[selected_features], y.values, cv=10,scoring = 'neg_mean_squared_log_error',
                         return_train_score=False)
-np.mean(cv_results_xgb['test_score'].tolist())


# In[ ]:


y_predict = xgb_model.predict(X_test[selected_features])
y_predict_xgb = y_predict.tolist()
X_test["SalePrice"] = y_predict.tolist()
final_data = X_test[["Id","SalePrice"]]
final_data.to_csv("D:\\Kaggle\\Housing Prices\\predictions_boruta_xgb.csv", index=False)


# In[ ]:


from sklearn.ensemble import ExtraTreesRegressor
et_model = ExtraTreesRegressor(n_estimators=1000, n_jobs= 2,oob_score= True, max_features= 5,bootstrap= True, random_state=42)
et_model.fit(X[selected_features], y.values)
et_model.oob_score_
cv_results_et = cross_validate(et_model, X[selected_features], y.values, cv=10,scoring = 'neg_mean_squared_log_error',
                         return_train_score=False)

-np.mean(cv_results_et['test_score'].tolist())


# In[ ]:


y_predict = et_model.predict(X_test[selected_features])
y_predict_et = y_predict.tolist()
X_test["SalePrice"] = y_predict.tolist()
final_data = X_test[["Id","SalePrice"]]
final_data.to_csv("D:\\Kaggle\\Housing Prices\\predictions_boruta_et.csv", index=False)


# In[ ]:


avg_pred_from_all_models = pd.DataFrame({'ID': X_test["Id"]})
avg_pred_from_all_models["rf"] = y_predict_rf
avg_pred_from_all_models["gbm"] = y_predict_gbm
avg_pred_from_all_models["xgb"] = y_predict_xgb
avg_pred_from_all_models["et"] = y_predict_et
avg_pred_from_all_models['SalePrice'] = avg_pred_from_all_models.drop(columns = "ID").mean(axis=1)
avg_pred_from_all_models
avg_pred_from_all_models[["ID","SalePrice"]].to_csv("D:\\Kaggle\\Housing Prices\\predictions_boruta_avg.csv", index=False)
