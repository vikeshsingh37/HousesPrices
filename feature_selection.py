from sklearn.ensemble import RandomForestRegressor
from boruta import BorutaPy as bp

class feature_selection():
	def __init__(self, X,y):
		self.X = X
		self.y = y

	def boruta_fs(self, n_estimators, max_iter, n_jobs):
		rf_model = RandomForestRegressor (n_jobs= n_jobs,oob_score= True)
		feat_selector = bp(rf_model,n_estimators = n_estimators, verbose= 1,max_iter= max_iter)
		feat_selector.fit(self.X.values, self.y.values)
		selected_features = self.X.columns[feat_selector.support_].tolist()
		return(selected_features)
