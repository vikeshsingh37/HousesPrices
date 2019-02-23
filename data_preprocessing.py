# This script is a common data processing for the Kaggle's housing competition

import sys
import numpy as np
import pandas as pd
from numpy import nanmedian
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import skew
from scipy.special import boxcox1p
from pyod.models.knn import KNN 

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
            one_hot_enc = OneHotEncoder(categories='auto')
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
        
        #These columns actually have NA values. So, these NAs shall be treated as NotAvailable
        cols_with_actual_NA = ['Alley', 'FireplaceQu','PoolQC' , 'Fence' ,'MiscFeature','BsmtQual','BsmtCond','BsmtExposure','FireplaceQu',
                              'GarageQual','GarageCond','PoolQC']
        cols_to_keep = list(set(cols_to_keep + cols_with_actual_NA))
        X = X[cols_to_keep]
        X[cols_with_actual_NA] = X[cols_with_actual_NA].fillna('NotAvailable')
        
        # These are actually categorical columns
        # 'MSSubClass' had some new categories that are absent in training. So, a hack here :)        
        #psuedo_num_cols = ['MSSubClass','OverallQual','OverallCond','YrSold','MoSold', 'YearRemodAdd','YearBuilt','GarageYrBlt']
        psuedo_num_cols = ['MSSubClass','OverallQual','OverallCond','GarageYrBlt']
        X[psuedo_num_cols] =X[psuedo_num_cols].fillna(X[psuedo_num_cols].median()[0])
        X['MSSubClass'] = X['MSSubClass'].map({20:1,30:2,40:3,45:4,50:5,60:6,70:7,75:8,80:9,85:10,90:11,120:12,150:13,160:14,180:15,190:16})  
        X['OverallQual'] = X['OverallQual'].map({10:10,9:9,8:8,7:7,6:6,5:5,4:4,3:3,2:2,1:1})  
        X['OverallCond'] = X['OverallCond'].map({10:10,9:9,8:8,7:7,6:6,5:5,4:4,3:3,2:2,1:1})
        #GarageYrBlt has NA values. Imputing by mode
        X['GarageYrBlt'] =X['GarageYrBlt'].fillna(X['GarageYrBlt'].mode()[0])
        #X[psuedo_num_cols] = X[psuedo_num_cols].applymap(str) 
        
        
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
        remaining_missing_value_columns.remove('GarageYrBlt')
        X = self.impute_data_using_train_data(X_train,X,remaining_missing_value_columns)

        # Check if still there are missing values
        nan_check = X[remaining_missing_value_columns].isnull().values.any()

        if(nan_check):
            print("There are still NA values")
            sys.exit()
            
        numeric_feats = X.dtypes[X.dtypes != "object"].index
        
        # Remove outliers
        clf = KNN()
        clf.fit(np.array(X.drop(cat_cols, axis = 1)))
        y_train_pred = clf.labels_.tolist()

        outlier_index = [i for i,val in enumerate(y_train_pred) if val ==1]
        X = X.drop(X.index[outlier_index])
        y = y.drop(y.index[outlier_index])
        X.reset_index(drop=True,inplace = True)
        y.reset_index(drop=True,inplace =True)
        
        print(str(len(outlier_index)) + " rows removed in training data as outliers")

        # Fix skewed numerical features
        skewed_feats = X[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
        skew_val = pd.DataFrame({'Skew' :skewed_feats})
        skew_val = skew_val[abs(skew_val.Skew) > 0.75]
        skewed_features = skew_val.index
        lam = 0.15
        for sf in skewed_features:
            X[sf] = boxcox1p(X[sf], lam)
            
        y = np.log1p(y)
        

        # Columns with order significance
        X['ExterQual'] = X['ExterQual'].map({'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0})
        X['ExterCond'] = X['ExterCond'].map({'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0})
        X['BsmtQual'] =  X['BsmtQual'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1,'NotAvailable': 0})
        X['BsmtCond'] = X['BsmtCond'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1,'NotAvailable': 0})
        X['BsmtExposure'] = X['BsmtExposure'].map({'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1,'NotAvailable': 0})
        X['HeatingQC'] = X['HeatingQC'].map({'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0})
        X['KitchenQual'] = X['KitchenQual'].map({'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0})
        X['FireplaceQu'] = X['FireplaceQu'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1,'NotAvailable': 0}) 
        X['GarageQual'] =  X['GarageQual'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1,'NotAvailable': 0}) 
        X['GarageCond'] = X['GarageCond'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1,'NotAvailable': 0})
        X['PoolQC'] = X['PoolQC'].map({'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'NotAvailable': 0})

        # Columns that need one hot encoding and have no ordering significance
        cols_to_one_hot_enc = ['MSZoning','Alley','LotShape','LandContour','LotConfig','LandSlope','Neighborhood',
                               'Condition1','BldgType','HouseStyle','RoofStyle','Exterior1st','Exterior2nd',
                               'MasVnrType','Foundation','BsmtFinType1','BsmtFinType2','Heating','CentralAir','Electrical',
                               'Functional','GarageType','GarageFinish','PavedDrive','Fence','MiscFeature','SaleType','SaleCondition'] #+ psuedo_num_cols
        
        # Convert string columns to one hot encoded vector
        X = self.encode_label_categorical_cols(training_data = X_train, new_data = X, cols_to_encode=cols_to_one_hot_enc)

        # Prepare test data for predictions
        columnsTitles = train_data.columns.tolist()
        columnsTitles.remove("SalePrice")
        test_data = test_data[columnsTitles]
        test_data = test_data.reindex(columns=columnsTitles)
        X_test = test_data
        
        X_test.drop(columns = high_imb_cats, axis = 1, inplace = True)
        
        # Data Processing as done with training data
        X_test[cols_with_actual_NA] = X_test[cols_with_actual_NA].fillna('NotAvailable')
        X_test[psuedo_num_cols]=X_test[psuedo_num_cols].fillna(X_test[psuedo_num_cols].median()[0])
        X_test['MSSubClass'] = X_test['MSSubClass'].map({20:1,30:2,40:3,45:4,50:5,60:6,70:7,75:8,80:9,85:10,90:11,120:12,150:13,160:14,180:15,190:16})  
        X_test['OverallQual'] = X_test['OverallQual'].map({10:10,9:9,8:8,7:7,6:6,5:5,4:4,3:3,2:2,1:1})  
        X_test['OverallCond'] = X_test['OverallCond'].map({10:10,9:9,8:8,7:7,6:6,5:5,4:4,3:3,2:2,1:1})
        #X_test['GarageYrBlt'] =X_test['GarageYrBlt'].fillna(X_test['GarageYrBlt'].mode()[0])
        #X_test[psuedo_num_cols] = X_test[psuedo_num_cols].applymap(str)
        X_test = self.impute_data_using_train_data(X_train,X_test,columnsTitles)
        
        for sf in skewed_features:
            X_test[sf] = boxcox1p(X_test[sf], lam)

        X_test['ExterQual'] = X_test['ExterQual'].map({'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0})
        X_test['ExterCond'] = X_test['ExterCond'].map({'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0})
        X_test['BsmtQual'] =  X_test['BsmtQual'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1,'NotAvailable': 0})
        X_test['BsmtCond'] = X_test['BsmtCond'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1,'NotAvailable': 0})
        X_test['BsmtExposure'] = X_test['BsmtExposure'].map({'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1,'NotAvailable': 0})
        X_test['HeatingQC'] = X_test['HeatingQC'].map({'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0})
        X_test['KitchenQual'] = X_test['KitchenQual'].map({'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0})
        X_test['FireplaceQu'] = X_test['FireplaceQu'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1,'NotAvailable': 0}) 
        X_test['GarageQual'] =  X_test['GarageQual'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1,'NotAvailable': 0}) 
        X_test['GarageCond'] = X_test['GarageCond'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1,'NotAvailable': 0})
        X_test['PoolQC'] = X_test['PoolQC'].map({'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'NotAvailable': 0})
        # Encode categorical data
        X_test = self.encode_label_categorical_cols(training_data = X_train, new_data = X_test, cols_to_encode=cols_to_one_hot_enc)

        return(X,X_test,y)