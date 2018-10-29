# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 18:07:04 2018

@author: pablo
"""

import pandas as pd 
import numpy as np

dfs = {}
origins = ['adwords',
           'facebook_ads',
           'sales',
           'analytics']

for df in origins:
    dfs[df] =  pd.read_excel('data_trial.xlsx',df)

# clean unused variables
"""
clean_df = {}    
for df in origins:
    clean_df[df]=dfs[df].loc[:,dfs[df].apply(pd.Series.nunique) != 1]
"""
# merge all dataframes 
pivot = dfs['sales'] # pivot dataframe

# adwords dataframe 
adwords =  dfs['adwords']
adwords['year'] = adwords['Week (Mon-Sun)'].dt.year
adwords['weeknumber'] = adwords['Week (Mon-Sun)'].dt.week  # adwords channels
# pivoting table 
adwords_pivoted =  pd.pivot_table(adwords,
                                  index=["year","weeknumber"],
                                  #values=["Price"],
                                  columns=["Advertising channel type",'Network'],
                                  #aggfunc=[np.sum]
                                  )

data=pd.merge(left= pivot,
               right =  adwords_pivoted,
               on = ['year','weeknumber'],
               how='inner'
               )
# unnest column names 
data.columns = [col for col in data.columns if type(col)==str] +  \
                ['_'.join(tup).rstrip('_') for tup in data.columns.values if type(tup)==tuple] 
# facebook dataframe
"""
facebook =  dfs['facebook_ads']
facebook['year'] = facebook['Week'].dt.year
facebook['weeknumber'] = facebook['Week'].dt.week  # adwords channels
"""

# ================================= #
#   detect correlation in columns   #
# ================================= #
# remove constant columns
data = data.loc[:,data.apply(pd.Series.nunique) != 1]
# list all mandatory variables
inv_columns = [col for col in data.columns if 'Cost_' in col]
keys = ['year','weeknumber'] # df keys 
target = ['total_paid'] # target variable
to_keep = inv_columns + keys + target

# remove high correlated variables 
# Create correlation matrix
corr_matrix = data.corr().abs()
# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
# Find features with correlation greater than treshold 
correlation_threshold=0.65
to_drop = [column for column in upper.columns \
           if any(upper[column] > correlation_threshold) and \
           column not in to_keep] + ['total_orders']
# Drop features 
data_sel_col =  data[[col for col in data.columns if col not in to_drop]]
# remove days without sale

#data_sel_col = data_sel_col.drop(data_sel_col[data_sel_col[target[0]]<=0].index)

# ============================ #
# === create shift columns === #
# ============================ #

# invesment lag 
data_sel_col[[col+'lag_1' for col in inv_columns]] = data_sel_col[inv_columns].shift(1)
data_sel_col[[col+'lag_2' for col in inv_columns]] = data_sel_col[inv_columns].shift(2)
data_sel_col[[col+'lag_3' for col in inv_columns]] = data_sel_col[inv_columns].shift(3)
# remove nan rows 
data_sel_col = data_sel_col.dropna(axis='rows')

# ============================ #
# ==== add seasonal vars ===== #
# ============================ #

#data_sel_col.total_paid.plot(subplots=True)
#data_sel_col.total_discounts.plot(subplots=True)
#a = data_sel_col[['year', 'weeknumber', 'total_discounts']]\
#                .sort_values('total_discounts')

# a = data_sel_col[['year', 'weeknumber', 'total_paid']]\
#                .sort_values('total_paid')

# cyber weeks
cyber = [(2018,21),
         (2017,22),
         (2016,22),
         (2018,40),
         (2017,45),
         (2016,45)]

data_sel_col.loc[:,'seasonal_cyber'] = np.where(data_sel_col[['year', 
                                                        'weeknumber']]\
                                             .apply(tuple, axis=1)\
                                             .isin(cyber),1,0)
# navidad 
# liquidaciones (total_discount var)
# ventas nocturnas
# web downtime 
# change sales validation
#data_sel_col[target[0]] = np.sqrt(data_sel_col[target[0]]) 
#data_sel_col[inv_columns] = np.log(data_sel_col[inv_columns])

# ================================ #
# ======== regression model ====== #
# ================================ #
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
# modelos lineales (incluyen la capacidad de restringir los signos de los parametros)
from sklearn.linear_model import LassoCV  # cross-validation param alpha
from sklearn.linear_model import LassoLarsCV  # idem with LARS
from sklearn.linear_model import LassoLarsIC  # using AIC/BIC en vez de alpha
from sklearn.linear_model import ElasticNetCV  # iteration over alpha, rho
from sklearn.linear_model import LarsCV
from sklearn.linear_model import ARDRegression
from sklearn.linear_model import BayesianRidge


models = [{'name':'LCV', 'mdl':LassoCV(n_alphas=100)},
          {'name':'LLCV','mdl':LassoLarsCV(max_n_alphas=1000)},
          {'name':'LLaic','mdl':LassoLarsIC(criterion='aic')},
          {'name':'ENCV', 'mdl':ElasticNetCV(n_alphas=100)},
          {'name':'LarsCV', 'mdl':LarsCV(max_n_alphas=1000)}, 
          {'name':'LR', 'mdl':LinearRegression()},
          {'name':'ARDR', 'mdl':ARDRegression()},
          {'name':'BYR', 'mdl':BayesianRidge()},
        ]

indep_vars = [col for col in data_sel_col if col not in target + keys] 
y_vector = data_sel_col[target]
x_matrix = data_sel_col[indep_vars].values

# ignore warnings 
import warnings
warnings.filterwarnings("ignore")

for mdl in models:
    r_square = model_selection.cross_val_score(mdl['mdl'],
                                               x_matrix,
                                               y_vector,
                                               scoring='r2',
                                               cv=5)
    
    final_r_square = r_square[r_square > 0].mean()
    mdl['score']=final_r_square
    print("{}, r2_score = {}".format(mdl['name'] , final_r_square))

# train best model*
#model = max(models, key=lambda mdl: mdl['score'])
import matplotlib.pyplot as plt
grid_number_columns = 4    
fig, axs = plt.subplots(int(len(models)/grid_number_columns),
                        grid_number_columns , 
                        constrained_layout=True)

# colors to channels
colormap = plt.cm.gist_ncar #nipy_spectral, Set1,Paired   
colors = [colormap(i) for i in np.linspace(0, 1,len(inv_columns))]

for i,model in enumerate(models):
    #model =  models[5]
    #fit model
    # predict with diferent inversions
    labels = []
    # by default all models qualify 
    
    model['qualify']=True 
    model['mdl'].fit(x_matrix,y_vector)
    for col_ix,inv_col in enumerate(inv_columns):
        
        n_sample = 100
        predict = pd.DataFrame(np.repeat(data_sel_col.iloc[-1:].values,n_sample,axis=0), 
                             columns = data_sel_col.columns)
        predict[inv_col] =  np.linspace(start = data_sel_col[inv_col].min(),
                                        stop = 1500000, #data_sel_col[inv_col].max(), 
                                        num = n_sample)
        
        predict['y_predicted_{}'.format(inv_col)]  =  model['mdl'].predict(predict[indep_vars].values)
        
        # qualify model 
        # based on the growth curve
        if not predict['y_predicted_{}'.format(inv_col)].is_monotonic:
             model['qualify']=False   
        
        #sns.scatterplot(x=inv_col, y="y_predicted", data=predict)
     
        #sns.lineplot(x = inv_col, 
        #             y="y_predicted_{}".format(inv_col)], 
        #             data=predict, 
        #             ax=ax,
        #             legend='full')
        labels.append(inv_col)    
        fig.axes[i].plot(predict[inv_col], 
                         predict['y_predicted_{}'.format(inv_col)], 
                         color=colors[col_ix])
        
    #fig.axes[i].legend(labels, ncol=4, loc='upper center', 
    #           bbox_to_anchor=[0.5, 1.1], 
    #           columnspacing=1.0, labelspacing=0.0,
    #           handletextpad=0.0, handlelength=1.5,
    #           fancybox=True, shadow=True)
    fig.axes[i].set_title('{} qualify={}'.format(model['name'],model['qualify'])  )
    fig.axes[i].set_xlabel('r2_score = {0:.3f}'.format(model['score']))
fig.show()

# ============================== # 
# ===== model assembly  ======== # 
# ============================== # 

# Model assembly based on score performance and convexity conditions over the 
# model curve
class AssembleModel():
    
    def __init__(self, models):
        selected_models = []
        for mdl in models:
            if mdl['qualify'] and mdl['score'] != np.nan:
               selected_models.append(mdl)
        self.selected_models = selected_models
        
    def predict(self, x):
        # given an x ndarray return convex prediction based on score of all 
        # qualified models 
        
        return sum([mdl['mdl'].predict(x) * mdl['score'] 
                    for mdl in self.selected_models])\
                /sum([mdl['score'] for mdl in self.selected_models])
        

best_model = AssembleModel(models)
best_model.predict(x_matrix)

# export model 
import dill
with open("best_model.pkl", "wb") as dill_file:
    dill.dump(best_model, dill_file)
    

# ======================================= # 
# ========= optimization model ========== #
# ======================================= #


from pulp import LpProblem, LpMaximize

# problem definition 
prob = LpProblem("MMM",LpMaximize)


for inv_var in inv_columns:
LpVariable("example", upBound = 100)
    
    
    