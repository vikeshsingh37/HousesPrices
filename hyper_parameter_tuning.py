# Author: Vikesh Singh Baghel
# Date: 16-Feb-2019
import tpot as tp

def tune_ga(model_dict,params,X,y,generations= 20, population_size=100, offspring_size=100, verbosity=2, early_stop=5,n_jobs= 4, 
                                    cv=5, scoring="neg_mean_squared_log_error"):
    
    ga_regressor = tp.TPOTRegressor(generations=generations, population_size=population_size, offspring_size = offspring_size, verbosity = verbosity,
                                     early_stop =early_stop,n_jobs =n_jobs, cv=cv, scoring=scoring,config_dict={model_dict: params})
    ga_regressor.fit(X, y)

    args = {}
    for arg in ga_regressor._optimized_pipeline:
        if type(arg) != 'Primitive':
            try:
                if arg.value.split('__')[1].split('=')[0] in ['max_depth', 'n_estimators', 'nthread','min_child_weight','n_jobs','random_state']:
                    args[arg.value.split('__')[1].split('=')[0]] = int(arg.value.split('__')[1].split('=')[1])
                else:
                    args[arg.value.split('__')[1].split('=')[0]] = float(arg.value.split('__')[1].split('=')[1])
            except:
                pass
    return(args)
