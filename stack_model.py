# Author: Vikesh Singh Baghel
# Date: 16-Feb-2019
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import pandas as pd

class stack_model:
    def __init__(self,base_models,meta_model,train_X,train_y):
        self.base_models = base_models
        self.meta_model  = meta_model
        self.train_X = train_X
        self.train_y = train_y
        
    def fit(self,cvFolds = 5, random_state= 42):
        meta_model_data = pd.DataFrame()
        for i,val in enumerate(self.base_models):
            meta_model_data[str(i)] = None
        meta_model_data["actual"] = None

        # Create kfolds
        kf = KFold(n_splits= cvFolds, shuffle=False, random_state=random_state)

        for train_index, test_index in kf.split(self.train_X):
            X_train, X_test = self.train_X.iloc[train_index], self.train_X.iloc[test_index]
            y_train, y_test = self.train_y[train_index], self.train_y[test_index]

            # Get predictions from all the base models
            this_fold_df = pd.DataFrame()
            for i,val in enumerate(self.base_models):
                self.base_models[i].fit(X_train,y_train)
                this_fold_df[str(i)] = self.base_models[i].predict(X_test)

            this_fold_df["actual"] = y_test.tolist()
            meta_model_data = meta_model_data.append(this_fold_df)

        # Build metaestimator model
        self.meta_model.fit(meta_model_data.drop("actual", axis =1),meta_model_data["actual"])
        

    def predict(self,test_X):
        meta_X = pd.DataFrame()
        for i,val in enumerate(self.base_models):
            self.base_models[i].fit(self.train_X,self.train_y)
            meta_X[str(i)] = self.base_models[i].predict(test_X)
        return(self.meta_model.predict(meta_X).tolist())