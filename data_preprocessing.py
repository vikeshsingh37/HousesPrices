# This script is a common data processing for the Kaggle's housing competition

import sys
import numpy as np
import pandas as pd
from numpy import nanmedian
from sklearn.preprocessing import OneHotEncoder

class data_preprocessing:
	def __init__(self,train_data_file, test_data_file):
		self.train_data_file = train_data_file
		self.test_data_file = test_data_file

	def impute_data_using_train_data(self, training_data, new_data, cols_to_impute):

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
	def encode_label_categorical_cols(self, training_data, new_data, cols_to_encode):
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

	def get_processed_data(self):
		train_data = pd.read_csv(self.train_data_file)
		test_data = pd.read_csv(self.test_data_file)
		train_data = train_data.sample(frac=1).reset_index(drop=True)
		X = train_data.drop(columns = ["SalePrice","Id"])
		y = train_data['SalePrice']

		# Find columns with missing values
		missing_value_columns = X.isnull().sum(axis = 0)
		good_columns = missing_value_columns[missing_value_columns == 0]
		missing_value_columns =  missing_value_columns[missing_value_columns > 0]
		miss_cols_precent = 100*missing_value_columns/X.shape[0]

		# High missing values are spread all over the neighborhood. So, removing them from feature selection. Rest may be imputed
		# Details on reaching this conclusion can be found in random_forest_with_boruta.py
		remaining_missing_value_columns = miss_cols_precent[miss_cols_precent < 40]
		remaining_missing_value_columns = remaining_missing_value_columns.axes[0].tolist()
		cols_to_keep = good_columns.axes[0].tolist() + remaining_missing_value_columns
		X = X[cols_to_keep]

		# Check the frequencies of character column. If the frequencies are close to number of training rows or there is huge imbalance
		# in categories these will be of no use in model building
		# Details on it can be found in random_forest_with_boruta.py
		cat_cols = X.select_dtypes(include = ["category","object"]).columns.tolist()

		# Columns to remove with high imbalance in categories
		high_imb_cats = ['RoofMatl', 'Condition2', 'Utilities', 'Street']
		X.drop(columns = high_imb_cats, axis = 1, inplace = True)


		cat_cols = [i for i in cat_cols if i not in high_imb_cats]

		# Impute missing numerical columns by median and categorical by most frequent
		X_train = X
		X = self.impute_data_using_train_data(X_train,X,remaining_missing_value_columns)

		# Check if still there are missing values
		nan_check = X[remaining_missing_value_columns].isnull().values.any()

		if(nan_check):
		    print("There are still NA values")
		    sys.exit()

		# Convert string columns to one hot encoded vector
		X = self.encode_label_categorical_cols(training_data = X_train, new_data = X, cols_to_encode=cat_cols)

		# Prepare test data for predictions
		columnsTitles = train_data.columns.tolist()
		columnsTitles.remove("SalePrice")
		test_data = test_data[columnsTitles]
		test_data = test_data.reindex(columns=columnsTitles)
		X_test = test_data
		# Impute data from training data
		X_test = self.impute_data_using_train_data(X_train,X_test,columnsTitles)

		# Encode categorical data
		X_test = self.encode_label_categorical_cols(training_data = X_train, new_data = X_test, cols_to_encode=cat_cols)

		return(X,X_test,y)
