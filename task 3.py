#!/usr/bin/env python
# coding: utf-8

# In[82]:


from sklearn.pipeline import make_pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np


# In[83]:


train_df = pd.read_csv('internship_train.csv')


# In[84]:


X, y = train_test_split(train_df, test_size=0.2, random_state=42)


# In[85]:


X_train = X.drop('target', axis=1)
y_train = X['target']
X_test = y.drop('target', axis=1)
y_test = y['target']


# In[86]:


# Define the preprocessing pipeline
preprocessor = make_pipeline(StandardScaler())


# In[87]:


# Train and evaluate different models
best_models = {}
for name, model in models:
    clf = make_pipeline(preprocessor, model)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print(f"{name} RMSE: {rmse}")


# In[88]:


if name == 'Decision Tree':
        param_grid = {'decisiontreeregressor__max_depth': [None, 10, 20, 30],
                      'decisiontreeregressor__min_samples_split': [2, 5, 10]}
        grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

elif name == 'XGBoost':
        param_grid = {'xgbregressor__n_estimators': [100, 500, 1000],
                      'xgbregressor__learning_rate': [0.01, 0.1, 0.5]}
        grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

elif name == 'Linear Regression':
    model = make_pipeline(StandardScaler(with_mean=False), LinearRegression())
    param_grid = {'linearregression__fit_intercept': [True, False]}
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

best_models[name] = best_model


# In[89]:


for name, model in best_models.items():
    model.fit(X_train, y_train)


# In[90]:


test_df = pd.read_csv('internship_hidden_test.csv')
y_pred_test = model.predict(test_df)


# In[91]:


predictions_df = pd.DataFrame({'target': y_pred_test})
predictions_df.to_csv('internship_test_predictions.csv', index=False)


# In[ ]:




