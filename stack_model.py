# Author: Vikesh Singh Baghel
# Date: 16-Feb-2019
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

def model_stacking(base_models,meta_model,train_X,train_y,testing_x = None,validation_percentage= 0.2, metrics = r2_score, cvFolds = 5, random_state= 42):
	
	# List to store data for meta_model training
	pred_x = []
	pred_y = []
	
	# Separate out validation set
	if(testing_x is None):
		train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size=validation_percentage, random_state=random_state)

	# Create kfolds
	kf = KFold(n_splits= cvFolds, shuffle=False, random_state=random_state)

	meta_score = []
	for train_index, test_index in kf.split(train_X):
	    X_train, X_test = train_X.iloc[train_index], train_X.iloc[test_index]
	    y_train, y_test = train_y[train_index], train_y[test_index]

	   	# Get predictions from all the base models
	    for model in base_models:
	    	model.fit(X_train,y_train)
	    	pred_x.append(model.predict(X_test))
	    	pred_y.append(y_test)

	# Build metaestimator model
	meta_model.fit(pred_x,pred_y)

	# Predict on validation or testing set
	if(testing_x is None):
		return(metrics(val_y,meta_model.predict(val_X)))
	else:
		return(meta_model.predict(testing_x))