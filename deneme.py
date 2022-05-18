import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import math
st.title('DirectLDL')
df = pd.read_csv("veri2zscore.csv", index_col=0)
if st.checkbox('Show dataframe'):
    st.write(df)
st.subheader('Scatter plot')
species = st.multiselect('Show iris per variety?', df['ZLDL_1'].unique())
col1 = st.selectbox('Which feature on x?', df.columns[0:4])
col2 = st.selectbox('Which feature on y?', df.columns[0:4])
new_df = df[(df['ZLDL_1'].isin(species))]
st.write(new_df)
# create figure using plotly express
fig = px.scatter(new_df, x =col1,y=col2, color='ZLDL_1')
# Plot!
st.plotly_chart(fig)
st.subheader('Histogram')
feature = st.selectbox('Which feature?', df.columns[0:4])
# Filter dataframe
new_df2 = df[(df['ZLDL_1'].isin(species))][feature]
fig2 = px.histogram(new_df, x=feature, color="ZLDL_1", marginal="rug")
st.plotly_chart(fig2)
st.subheader('Machine Learning models')
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
import pandas
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, cross_validate,cross_val_score, cross_val_predict
from sklearn.utils import shuffle
# Necessary imports:

from sklearn import metrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

features= df.drop(['ZLDL_1'],axis=1)
labels = df['ZLDL_1']
st.write(features)
X_train,X_test, y_train, y_test = train_test_split(features, labels, train_size=0.7, random_state=1)
alg = ['Decision Tree', 'Support Vector Regression','Linear Regression']
classifier = st.selectbox('Which algorithm?', alg)
if classifier=='Decision Tree':
    dtc = DecisionTreeRegressor()
    dtc.fit(X_train, y_train)
    acc = dtc.score(X_test, y_test)
    st.write('Accuracy: ', acc)
    pred_dtc = dtc.predict(X_test)
    #cm_dtc=confusion_matrix(y_test,pred_dtc)
    #st.write('Confusion matrix: ', cm_dtc)
elif classifier == 'Support Vector Regression':
    """svm=SVR()
    svm.fit(X_train, y_train)
    acc = svm.score(X_test, y_test)
    st.write('Accuracy: ', acc)
    pred_svm = svm.predict(X_test)"""
    gsc = GridSearchCV(
        estimator=SVR(kernel='rbf'),
        param_grid={
            'C': [0.1, 1, 100, 1000],
            'epsilon': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10],
            'gamma': [0.0001, 0.001, 0.005, 0.1, 1, 3, 5]
        },
        cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)

    grid_result = gsc.fit(X_train, y_train)
    best_params = grid_result.best_params_
    st.write("best_params",best_params)
    best_svr = SVR(kernel='rbf', C=best_params["C"], epsilon=best_params["epsilon"], gamma=best_params["gamma"],
                   coef0=0.1, shrinking=True,
                   tol=0.001, cache_size=200, verbose=False, max_iter=-1)

    scoring = {
               'abs_error': 'neg_mean_absolute_error',
               'squared_error': 'neg_mean_squared_error'}

    scores = cross_validate(best_svr, X_train, y_train, cv=10, scoring=scoring, return_train_score=True)
    #print(scores)
    predictions = cross_val_predict(best_svr, X_train, y_train, cv=5)
    st.write(predictions.shape)
    st.write(y_train.shape)
    accuracy = metrics.r2_score(y_train, predictions)
    st.write ("Cross-Predicted Accuracy:", accuracy)
    plt.scatter(y_train, predictions)
    st.write("MAE :", abs(scores['test_abs_error'].mean()), "| RMSE :", math.sqrt(abs(scores['test_squared_error'].mean())))
    #cm=confusion_matrix(y_test,pred_svm)
    #st.write('Confusion matrix: ', cm)
elif classifier == 'Linear Regression':
    lr=LinearRegression()
    lr.fit(X_train, y_train)
    acc = lr.score(X_test, y_test)
    st.write('R2 Score: ', acc)
    pred_lr = lr.predict(X_test)
    ##rBF kernel

