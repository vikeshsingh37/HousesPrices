import sys
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.ensemble import RandomForestRegressor
from boruta import BorutaPy as bp
from sklearn.metrics import r2_score
from numpy import nanmedian
from sklearn.preprocessing import OneHotEncoder

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

# Load data
train_data = pd.read_csv("D:\\Kaggle\\Housing Prices\\train.csv")
test_data = pd.read_csv("D:\\Kaggle\\Housing Prices\\test.csv")
columnsTitles = train_data.columns.tolist()

X = train_data.drop(columns = ["SalePrice","Id"])
y = train_data['SalePrice']
test_data = test_data.reindex(columns=columnsTitles)

# Find columns with missing values
missing_value_columns = X.isnull().sum(axis = 0)
missing_value_columns =  missing_value_columns[missing_value_columns > 0]
miss_cols_precent = 100*missing_value_columns/X.shape[0]
print(miss_cols_precent)

# Remove columns with highly imputed values
remaining_missing_value_columns = X.isnull().sum(axis = 0)
remaining_missing_value_columns = remaining_missing_value_columns[remaining_missing_value_columns > 0]
remaining_missing_value_columns = remaining_missing_value_columns.axes[0].tolist()
print(remaining_missing_value_columns)

# Impute missing numerical columns by median
X_train = X
X = impute_data_using_train_data(X_train,X,remaining_missing_value_columns)

# Check if still there are missing values
nan_check = X.isnull().values.any()
print(nan_check)

if(nan_check):
    print("There are still NA values")
    sys.exit()

# Convert string columns to one hot encoded vector
cat_cols = X.select_dtypes(include = ["category","object"]).columns.tolist()
X = encode_label_categorical_cols(training_data = X_train, new_data = X, cols_to_encode=cat_cols)

# Feature selection by Boruta
rf_model = RandomForestRegressor(n_estimators=1000, oob_score= True,max_features  = 4)
feat_selector = bp(rf_model,n_estimators = 'auto', verbose= 1,max_iter= 1000)
feat_selector.fit(X.values, y.values)

# Build Random Forest Model
selected_features = X.columns[feat_selector.support_]
print(selected_features)
selected_features_data = X[selected_features]
rf_model.fit(selected_features_data, y.values)
rf_model.oob_score_

# Prepare test data for predictions
X_test = test_data
# Impute data from training data
X_test = impute_data_using_train_data(X_train,X_test,X_train.columns.tolist())

# Encode categorical data
columnsToEncode = list(X_test.select_dtypes(include=['category','object']))
X_test = encode_label_categorical_cols(training_data = X_train, new_data = X_test, cols_to_encode=columnsToEncode)

# Predict 
#y_predict = rf_model.predict(X_test[selected_features])
model = feat_selector.estimator
y_predict = model.predict(X_test[selected_features])

test_data["SalePrice"] = y_predict.tolist()
final_data = test_data[["Id","SalePrice"]]
final_data.head()
final_data.to_csv("D:\\Kaggle\\Housing Prices\\predictions.csv", index=False)
