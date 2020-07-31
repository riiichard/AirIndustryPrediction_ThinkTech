# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Thu Jul  9 10:46:55 2020

# @author: Richard
# """


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# from mpl_toolkits.axes_grid1.inset_locator import mark_inset
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# from sklearn.neural_network import MLPRegressor
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.tsa.api import ExponentialSmoothing

df = pd.read_csv("~/Desktop/普华永道/首都机场.csv",header = 0,index_col=0)
matrix = df.values
df_rever=df.T

amount = df_rever[['旅客吞吐量(人次)']]
amount.plot()
plt.title('首都机场旅客吞吐量')
plt.xlabel('月-年')
plt.ylabel('旅客吞吐量(人次)')
plt.savefig("首都机场旅客吞吐量.png")


# ======================================= Regression ==================================

amount_no_outlier = amount.loc[:'Dec-19']
x_axis = np.arange(amount_no_outlier.size) + 1
Xtrain, Xvalid, ytrain, yvalid = train_test_split(x_axis,amount_no_outlier,random_state=0,test_size=48)
ytrain, yvalid = ytrain.values[:,0], yvalid.values[:,0]
Xtrain = Xtrain.reshape((192,1))
Xvalid = Xvalid.reshape((48,1))

# # linear regression:
# # model = linear_model.Ridge(alpha=0.01)
# # model = linear_model.SGDRegressor(loss='huber', penalty='l1', alpha=0.0001)
# model = linear_model.LinearRegression()
# # model = linear_model.Lasso()
# model.fit(Xtrain, ytrain)
# ytrain_pred = model.predict(Xtrain)
# training_error = np.average((ytrain_pred - np.array(ytrain)) ** 2)
# yvalid_pred = model.predict(Xvalid)
# validation_error = np.average((yvalid_pred - np.array(yvalid)) ** 2)
# amount = df_rever[['旅客吞吐量(人次)']]
# plt.figure()
# plt.title('首都机场旅客吞吐量')
# plt.xlabel('月-年')
# plt.ylabel('旅客吞吐量(人次)')
# plt.plot(amount.values)
# plt.plot(Xtrain, ytrain_pred)
# plt.show()
# print(training_error / 10**12, validation_error / 10**12)
# # plt.savefig("首都机场旅客吞吐量.png")

# # polynomial regression degree-2:
# quadratic_featurizer = PolynomialFeatures(degree=2)
# X_train_quadratic = quadratic_featurizer.fit_transform(Xtrain)
# X_valid_quadratic = quadratic_featurizer.fit_transform(Xvalid)
# model = linear_model.Lasso()
# model.fit(X_train_quadratic, ytrain)
# xx = np.linspace(0, 240, 240)
# xx_quadratic = quadratic_featurizer.transform(xx.reshape(xx.shape[0], 1))
# yy_predict = model.predict(xx_quadratic)
# ytrain_pred = model.predict(X_train_quadratic)
# training_error = np.average((ytrain_pred - np.array(ytrain)) ** 2)
# yvalid_pred = model.predict(X_valid_quadratic)
# validation_error = np.average((yvalid_pred - np.array(yvalid)) ** 2)
# amount = df_rever[['旅客吞吐量(人次)']]
# plt.figure()
# plt.title('首都机场旅客吞吐量')
# plt.xlabel('月-年')
# plt.ylabel('旅客吞吐量(人次)')
# plt.plot(amount.values)
# plt.plot(xx, yy_predict)
# plt.show()
# print(training_error / 10**12, validation_error / 10**12)

# polynomial regression degree-3:
quadratic_featurizer = PolynomialFeatures(degree=3)
X_train_quadratic = quadratic_featurizer.fit_transform(Xtrain)
X_valid_quadratic = quadratic_featurizer.fit_transform(Xvalid)
# model = linear_model.Lasso(max_iter=10000)
model = linear_model.LinearRegression()
model.fit(X_train_quadratic, ytrain)
xx = np.linspace(0, 246, 246)
xx_quadratic = quadratic_featurizer.transform(xx.reshape(xx.shape[0], 1))
yy_predict = model.predict(xx_quadratic)
ytrain_pred = model.predict(X_train_quadratic)
training_error = np.average((ytrain_pred - np.array(ytrain)) ** 2)
yvalid_pred = model.predict(X_valid_quadratic)
validation_error = np.average((yvalid_pred - np.array(yvalid)) ** 2)
amount = df_rever[['旅客吞吐量(人次)']]
plt.figure()
plt.title('首都机场旅客吞吐量')
plt.xlabel('月-年')
plt.ylabel('旅客吞吐量(人次)')
plt.plot(amount.values)
plt.plot(xx, yy_predict)
plt.show()
print(training_error / 10**12, validation_error / 10**12)

# # polynomial regression degree-4:
# quadratic_featurizer = PolynomialFeatures(degree=4)
# X_train_quadratic = quadratic_featurizer.fit_transform(Xtrain)
# X_valid_quadratic = quadratic_featurizer.fit_transform(Xvalid)
# # model = linear_model.Lasso(max_iter=100000)
# model = linear_model.LinearRegression()
# model.fit(X_train_quadratic, ytrain)
# xx = np.linspace(0, 246, 246)
# xx_quadratic = quadratic_featurizer.transform(xx.reshape(xx.shape[0], 1))
# yy_predict = model.predict(xx_quadratic)
# ytrain_pred = model.predict(X_train_quadratic)
# training_error = np.average((ytrain_pred - np.array(ytrain)) ** 2)
# yvalid_pred = model.predict(X_valid_quadratic)
# validation_error = np.average((yvalid_pred - np.array(yvalid)) ** 2)
# amount = df_rever[['旅客吞吐量(人次)']]
# plt.figure()
# plt.title('首都机场旅客吞吐量')
# plt.xlabel('月-年')
# plt.ylabel('旅客吞吐量(人次)')
# plt.plot(amount.values)
# plt.plot(xx, yy_predict)
# plt.show()
# print(training_error / 10**12, validation_error / 10**12)

# # polynomial regression degree-5:
# quadratic_featurizer = PolynomialFeatures(degree=5)
# X_train_quadratic = quadratic_featurizer.fit_transform(Xtrain)
# X_valid_quadratic = quadratic_featurizer.fit_transform(Xvalid)
# # model = linear_model.Lasso(max_iter=100000)
# model = linear_model.LinearRegression()
# model.fit(X_train_quadratic, ytrain)
# xx = np.linspace(0, 246, 246)
# xx_quadratic = quadratic_featurizer.transform(xx.reshape(xx.shape[0], 1))
# yy_predict = model.predict(xx_quadratic)
# ytrain_pred = model.predict(X_train_quadratic)
# training_error = np.average((ytrain_pred - np.array(ytrain)) ** 2)
# yvalid_pred = model.predict(X_valid_quadratic)
# validation_error = np.average((yvalid_pred - np.array(yvalid)) ** 2)
# amount = df_rever[['旅客吞吐量(人次)']]
# plt.figure()
# plt.title('首都机场旅客吞吐量')
# plt.xlabel('月-年')
# plt.ylabel('旅客吞吐量(人次)')
# plt.plot(amount.values)
# plt.plot(xx, yy_predict)
# plt.show()
# print(training_error / 10**12, validation_error / 10**12)

# =====================================================================================





# ====================================== enlarge =======================================

amount = df_rever[['旅客吞吐量(人次)']]
domestic = df_rever[['国内航线(大陆)']]
foreign = df_rever[['国际航线(加港澳台地区)']]
covid_a = amount.loc['May-10':]
covid_d = domestic.loc['May-10':]
covid_i = foreign.loc['May-10':]
plt.figure()
plt.title('首都机场旅客吞吐量_COVID')
plt.xlabel('月-年')
plt.ylabel('旅客吞吐量(人次)')
plt.plot(covid_a.values)
plt.plot(covid_d.values)
plt.plot(covid_i.values)
plt.legend(['Total','Mainland','Inter+HMT'])
plt.show()
plt.savefig("首都机场旅客吞吐量_COVID.png")

# amount = df_rever[['旅客吞吐量(人次)']]
# domestic = df_rever[['国内航线(大陆)']]
# foreign = df_rever[['国际航线(加港澳台地区)']]
# # amount.plot(figsize=(20,10))
# sars_a = amount.loc['May-01':'May-05']
# sars_d = domestic.loc['May-01':'May-05']
# sars_i = foreign.loc['May-01':'May-05']
# plt.figure()
# plt.title('首都机场旅客吞吐量_SARS')
# plt.xlabel('月-年')
# plt.ylabel('旅客吞吐量(人次)')
# plt.plot(sars_a.values)
# plt.plot(sars_d.values)
# plt.plot(sars_i.values)
# plt.legend(['Total','Mainland','Inter+HMT'])
# plt.show()
# plt.savefig("首都机场旅客吞吐量.png")

amount = df_rever[['旅客吞吐量(人次)']]
domestic = df_rever[['国内航线(大陆)']]
foreign = df_rever[['国际航线(加港澳台地区)']]
# amount.plot(figsize=(20,10))
sars_a = amount.loc['May-00':'May-06']
sars_d = domestic.loc['May-00':'May-06']
sars_i = foreign.loc['May-00':'May-06']
plt.figure()
plt.title('首都机场旅客吞吐量_SARS')
plt.xlabel('月-年')
plt.ylabel('旅客吞吐量(人次)')
plt.plot(sars_a.values)
plt.plot(sars_d.values)
plt.plot(sars_i.values)
plt.legend(['Total','Mainland','Inter+HMT'])
plt.show()
plt.savefig("首都机场旅客吞吐量.png")

# ===================================================================================





# =============================== domestric with regression ==========================

domestic = df_rever[['国内航线(大陆)']]
domestic_no_outlier = domestic.loc[:'Dec-19']
x_axis = np.arange(domestic_no_outlier.size) + 1
Xtrain, Xvalid, ytrain, yvalid = train_test_split(x_axis,domestic_no_outlier,random_state=0,test_size=48)
ytrain, yvalid = ytrain.values[:,0], yvalid.values[:,0]
Xtrain = Xtrain.reshape((192,1))
Xvalid = Xvalid.reshape((48,1))

# # linear regression:
# # model = linear_model.Ridge(alpha=0.01)
# # model = linear_model.SGDRegressor(loss='huber', penalty='l1', alpha=0.0001)
# model = linear_model.LinearRegression()
# # model = linear_model.Lasso()
# model.fit(Xtrain, ytrain)
# ytrain_pred = model.predict(Xtrain)
# training_error = np.average((ytrain_pred - np.array(ytrain)) ** 2)
# yvalid_pred = model.predict(Xvalid)
# validation_error = np.average((yvalid_pred - np.array(yvalid)) ** 2)
# plt.figure()
# plt.title('首都机场大陆旅客吞吐量')
# plt.xlabel('月-年')
# plt.ylabel('旅客吞吐量(人次)')
# plt.plot(domestic.values)
# plt.plot(Xtrain, ytrain_pred)
# plt.show()
# print(training_error / 10**12, validation_error / 10**12)
# plt.savefig("首都机场大陆旅客吞吐量.png")

# # polynomial regression degree-2:
# quadratic_featurizer = PolynomialFeatures(degree=2)
# X_train_quadratic = quadratic_featurizer.fit_transform(Xtrain)
# X_valid_quadratic = quadratic_featurizer.fit_transform(Xvalid)
# model = linear_model.Lasso()
# model.fit(X_train_quadratic, ytrain)
# xx = np.linspace(0, 240, 240)
# xx_quadratic = quadratic_featurizer.transform(xx.reshape(xx.shape[0], 1))
# yy_predict = model.predict(xx_quadratic)
# ytrain_pred = model.predict(X_train_quadratic)
# training_error = np.average((ytrain_pred - np.array(ytrain)) ** 2)
# yvalid_pred = model.predict(X_valid_quadratic)
# validation_error = np.average((yvalid_pred - np.array(yvalid)) ** 2)
# domestic = df_rever[['国内航线(大陆)']]
# plt.figure()
# plt.title('首都机场大陆旅客吞吐量')
# plt.xlabel('月-年')
# plt.ylabel('旅客吞吐量(人次)')
# plt.plot(domestic.values)
# plt.plot(xx, yy_predict)
# plt.show()
# print(training_error / 10**12, validation_error / 10**12)

# polynomial regression degree-3:
quadratic_featurizer = PolynomialFeatures(degree=3)
X_train_quadratic = quadratic_featurizer.fit_transform(Xtrain)
X_valid_quadratic = quadratic_featurizer.fit_transform(Xvalid)
# model = linear_model.Lasso(max_iter=10000)
model = linear_model.LinearRegression()
model.fit(X_train_quadratic, ytrain)
xx = np.linspace(0, 246, 246)
xx_quadratic = quadratic_featurizer.transform(xx.reshape(xx.shape[0], 1))
yy_predict = model.predict(xx_quadratic)
ytrain_pred = model.predict(X_train_quadratic)
training_error = np.average((ytrain_pred - np.array(ytrain)) ** 2)
yvalid_pred = model.predict(X_valid_quadratic)
validation_error = np.average((yvalid_pred - np.array(yvalid)) ** 2)
domestic = df_rever[['国内航线(大陆)']]
plt.figure()
plt.title('首都机场大陆旅客吞吐量')
plt.xlabel('月-年')
plt.ylabel('旅客吞吐量(人次)')
plt.plot(domestic.values)
plt.plot(xx, yy_predict)
plt.show()
print(training_error / 10**12, validation_error / 10**12)

# polynomial regression degree-4:
quadratic_featurizer = PolynomialFeatures(degree=4)
X_train_quadratic = quadratic_featurizer.fit_transform(Xtrain)
X_valid_quadratic = quadratic_featurizer.fit_transform(Xvalid)
# model = linear_model.Lasso(max_iter=100000)
model = linear_model.LinearRegression()
model.fit(X_train_quadratic, ytrain)
xx = np.linspace(0, 246, 246)
xx_quadratic = quadratic_featurizer.transform(xx.reshape(xx.shape[0], 1))
yy_predict = model.predict(xx_quadratic)
ytrain_pred = model.predict(X_train_quadratic)
training_error = np.average((ytrain_pred - np.array(ytrain)) ** 2)
yvalid_pred = model.predict(X_valid_quadratic)
validation_error = np.average((yvalid_pred - np.array(yvalid)) ** 2)
domestic = df_rever[['国内航线(大陆)']]
plt.figure()
plt.title('首都机场大陆旅客吞吐量')
plt.xlabel('月-年')
plt.ylabel('旅客吞吐量(人次)')
plt.plot(domestic.values)
plt.plot(xx, yy_predict)
plt.show()
print(training_error / 10**12, validation_error / 10**12)

# ===================================================================================

foreign = df_rever[['国际航线(加港澳台地区)']]
foreign_no_outlier = foreign.loc[:'Dec-19']
x_axis = np.arange(foreign_no_outlier.size) + 1
Xtrain, Xvalid, ytrain, yvalid = train_test_split(x_axis,foreign_no_outlier,random_state=0,test_size=48)
ytrain, yvalid = ytrain.values[:,0], yvalid.values[:,0]
Xtrain = Xtrain.reshape((192,1))
Xvalid = Xvalid.reshape((48,1))

# # linear regression:
# # model = linear_model.Ridge(alpha=0.01)
# # model = linear_model.SGDRegressor(loss='huber', penalty='l1', alpha=0.0001)
# model = linear_model.LinearRegression()
# # model = linear_model.Lasso()
# model.fit(Xtrain, ytrain)
# ytrain_pred = model.predict(Xtrain)
# training_error = np.average((ytrain_pred - np.array(ytrain)) ** 2)
# yvalid_pred = model.predict(Xvalid)
# validation_error = np.average((yvalid_pred - np.array(yvalid)) ** 2)
# plt.figure()
# plt.title('首都机场国际加港澳台旅客吞吐量')
# plt.xlabel('月-年')
# plt.ylabel('旅客吞吐量(人次)')
# plt.plot(foreign.values)
# plt.plot(Xtrain, ytrain_pred)
# plt.show()
# print(training_error / 10**12, validation_error / 10**12)
# # plt.savefig("首都机场国际加港澳台旅客吞吐量.png")

# polynomial regression degree-2:
quadratic_featurizer = PolynomialFeatures(degree=2)
X_train_quadratic = quadratic_featurizer.fit_transform(Xtrain)
X_valid_quadratic = quadratic_featurizer.fit_transform(Xvalid)
model = linear_model.Lasso()
model.fit(X_train_quadratic, ytrain)
xx = np.linspace(0, 240, 240)
xx_quadratic = quadratic_featurizer.transform(xx.reshape(xx.shape[0], 1))
yy_predict = model.predict(xx_quadratic)
ytrain_pred = model.predict(X_train_quadratic)
training_error = np.average((ytrain_pred - np.array(ytrain)) ** 2)
yvalid_pred = model.predict(X_valid_quadratic)
validation_error = np.average((yvalid_pred - np.array(yvalid)) ** 2)
foreign = df_rever[['国际航线(加港澳台地区)']]
plt.figure()
plt.title('首都机场国际加港澳台旅客吞吐量')
plt.xlabel('月-年')
plt.ylabel('旅客吞吐量(人次)')
plt.plot(foreign.values)
plt.plot(xx, yy_predict)
plt.show()
print(training_error / 10**12, validation_error / 10**12)

# # polynomial regression degree-3:
# quadratic_featurizer = PolynomialFeatures(degree=3)
# X_train_quadratic = quadratic_featurizer.fit_transform(Xtrain)
# X_valid_quadratic = quadratic_featurizer.fit_transform(Xvalid)
# # model = linear_model.Lasso(max_iter=10000)
# model = linear_model.LinearRegression()
# model.fit(X_train_quadratic, ytrain)
# xx = np.linspace(0, 246, 246)
# xx_quadratic = quadratic_featurizer.transform(xx.reshape(xx.shape[0], 1))
# yy_predict = model.predict(xx_quadratic)
# ytrain_pred = model.predict(X_train_quadratic)
# training_error = np.average((ytrain_pred - np.array(ytrain)) ** 2)
# yvalid_pred = model.predict(X_valid_quadratic)
# validation_error = np.average((yvalid_pred - np.array(yvalid)) ** 2)
# foreign = df_rever[['国际航线(加港澳台地区)']]
# plt.figure()
# plt.title('首都机场国际加港澳台旅客吞吐量')
# plt.xlabel('月-年')
# plt.ylabel('旅客吞吐量(人次)')
# plt.plot(foreign.values)
# plt.plot(xx, yy_predict)
# plt.show()
# print(training_error / 10**12, validation_error / 10**12)

# # polynomial regression degree-4:
# quadratic_featurizer = PolynomialFeatures(degree=4)
# X_train_quadratic = quadratic_featurizer.fit_transform(Xtrain)
# X_valid_quadratic = quadratic_featurizer.fit_transform(Xvalid)
# # model = linear_model.Lasso(max_iter=100000)
# model = linear_model.LinearRegression()
# model.fit(X_train_quadratic, ytrain)
# xx = np.linspace(0, 246, 246)
# xx_quadratic = quadratic_featurizer.transform(xx.reshape(xx.shape[0], 1))
# yy_predict = model.predict(xx_quadratic)
# ytrain_pred = model.predict(X_train_quadratic)
# training_error = np.average((ytrain_pred - np.array(ytrain)) ** 2)
# yvalid_pred = model.predict(X_valid_quadratic)
# validation_error = np.average((yvalid_pred - np.array(yvalid)) ** 2)
# foreign = df_rever[['国际航线(加港澳台地区)']]
# plt.figure()
# plt.title('首都机场国际加港澳台旅客吞吐量')
# plt.xlabel('月-年')
# plt.ylabel('旅客吞吐量(人次)')
# plt.plot(foreign.values)
# plt.plot(xx, yy_predict)
# plt.show()
# print(training_error / 10**12, validation_error / 10**12)

# ====================================================================================




# ====================================== Months =======================================

# Extract data for January
January=['Jan-00','Jan-01','Jan-02','Jan-03','Jan-04','Jan-05','Jan-06','Jan-07'
          ,'Jan-08','Jan-09','Jan-10','Jan-11','Jan-12','Jan-13','Jan-14','Jan-15'
          ,'Jan-16','Jan-17','Jan-18','Jan-19','Jan-20']
Jan = df_rever.loc[January]

# Extract data for February
February=['Feb-00','Feb-01','Feb-02','Feb-03','Feb-04','Feb-05','Feb-06','Feb-07'
          ,'Feb-08','Feb-09','Feb-10','Feb-11','Feb-12','Feb-13','Feb-14','Feb-15'
          ,'Feb-16','Feb-17','Feb-18','Feb-19','Feb-20']
Feb = df_rever.loc[February]

# Extract data for March
March=['Mar-00','Mar-01','Mar-02','Mar-03','Mar-04','Mar-05','Mar-06','Mar-07'
          ,'Mar-08','Mar-09','Mar-10','Mar-11','Mar-12','Mar-13','Mar-14','Mar-15'
          ,'Mar-16','Mar-17','Mar-18','Mar-19','Mar-20']
Mar = df_rever.loc[March]

# Extract data for April
April=['Apr-00','Apr-01','Apr-02','Apr-03','Apr-04','Apr-05','Apr-06','Apr-07'
          ,'Apr-08','Apr-09','Apr-10','Apr-11','Apr-12','Apr-13','Jan-14','Apr-15'
          ,'Apr-16','Apr-17','Apr-18','Apr-19','Apr-20']
Apr = df_rever.loc[April]

# Extract data for May
May=['May-00','May-01','May-02','May-03','May-04','May-05','May-06','May-07'
          ,'May-08','May-09','May-10','May-11','May-12','May-13','May-14','May-15'
          ,'May-16','May-17','Jan-18','Jan-19','May-20']
May = df_rever.loc[May]

# Extract data for June
June=['Jun-00','Jun-01','Jun-02','Jun-03','Jun-04','Jun-05','Jun-06','Jun-07'
          ,'Jun-08','Jun-09','Jun-10','Jun-11','Jun-12','Jun-13','Jun-14','Jun-15'
          ,'Jun-16','Jun-17','Jun-18','Jun-19','Jun-20']
Jun = df_rever.loc[June]

# Extract data for July
July=['Jul-00','Jul-01','Jul-02','Jul-03','Jul-04','Jul-05','Jul-06','Jul-07'
          ,'Jul-08','Jul-09','Jul-10','Jul-11','Jul-12','Jul-13','Jul-14','Jul-15'
          ,'Jul-16','Jul-17','Jul-18','Jul-19']
Jul = df_rever.loc[July]

# Extract data for August
August=['Aug-00','Aug-01','Aug-02','Aug-03','Aug-04','Aug-05','Aug-06','Aug-07'
          ,'Aug-08','Aug-09','Aug-10','Aug-11','Aug-12','Aug-13','Aug-14','Aug-15'
          ,'Aug-16','Aug-17','Aug-18','Aug-19']
Aug = df_rever.loc[August]

# Extract data for September
September=['Sep-00','Sep-01','Sep-02','Sep-03','Sep-04','Sep-05','Sep-06','Sep-07'
          ,'Sep-08','Sep-09','Sep-10','Sep-11','Sep-12','Sep-13','Sep-14','Sep-15'
          ,'Sep-16','Sep-17','Sep-18','Sep-19']
Sep = df_rever.loc[September]

# Extract data for October
October=['Oct-00','Oct-01','Oct-02','Oct-03','Oct-04','Oct-05','Oct-06','Oct-07'
          ,'Oct-08','Oct-09','Oct-10','Oct-11','Oct-12','Oct-13','Oct-14','Oct-15'
          ,'Oct-16','Oct-17','Oct-18','Oct-19']
Oct = df_rever.loc[October]

# Extract data for November
November=['Nov-00','Nov-01','Nov-02','Nov-03','Nov-04','Nov-05','Nov-06','Nov-07'
          ,'Nov-08','Nov-09','Nov-10','Nov-11','Nov-12','Nov-13','Nov-14','Nov-15'
          ,'Nov-16','Nov-17','Nov-18','Nov-19']
Nov = df_rever.loc[November]

# Extract data for December
December=['Dec-00','Dec-01','Dec-02','Dec-03','Dec-04','Dec-05','Dec-06','Dec-07'
          ,'Dec-08','Dec-09','Dec-10','Dec-11','Dec-12','Dec-13','Dec-14','Dec-15'
          ,'Dec-16','Dec-17','Dec-18','Dec-19']
Dec = df_rever.loc[December]

# Plot the month amounts of each year
Jan_amount = Jan[['旅客吞吐量(人次)']]
Jan_x = Jan_amount.values
Jan_d = Jan[['国内航线(大陆)']].values
Jan_i = Jan[['国际航线(加港澳台地区)']].values
Feb_amount = Feb[['旅客吞吐量(人次)']]
Feb_x = Feb_amount.values
Feb_d = Feb[['国内航线(大陆)']].values
Feb_i = Feb[['国际航线(加港澳台地区)']].values
Mar_amount = Mar[['旅客吞吐量(人次)']]
Mar_x = Mar_amount.values
Mar_d = Mar[['国内航线(大陆)']].values
Mar_i = Mar[['国际航线(加港澳台地区)']].values
Apr_amount = Apr[['旅客吞吐量(人次)']]
Apr_x = Apr_amount.values
Apr_d = Apr[['国内航线(大陆)']].values
Apr_i = Apr[['国际航线(加港澳台地区)']].values
May_amount = May[['旅客吞吐量(人次)']]
May_x = May_amount.values
May_d = May[['国内航线(大陆)']].values
May_i = May[['国际航线(加港澳台地区)']].values
Jun_amount = Jun[['旅客吞吐量(人次)']]
Jun_x = Jun_amount.values
Jun_d = Jun[['国内航线(大陆)']].values
Jun_i = Jun[['国际航线(加港澳台地区)']].values
Jul_amount = Jul[['旅客吞吐量(人次)']]
Jul_x = Jul_amount.values                 
Jul_d = Jul[['国内航线(大陆)']].values
Jul_i = Jul[['国际航线(加港澳台地区)']].values
Aug_amount = Aug[['旅客吞吐量(人次)']]
Aug_x = Aug_amount.values
Aug_d = Aug[['国内航线(大陆)']].values
Aug_i = Aug[['国际航线(加港澳台地区)']].values
Sep_amount = Sep[['旅客吞吐量(人次)']]
Sep_x = Sep_amount.values
Sep_d = Sep[['国内航线(大陆)']].values
Sep_i = Sep[['国际航线(加港澳台地区)']].values
Oct_amount = Oct[['旅客吞吐量(人次)']]
Oct_x = Oct_amount.values
Oct_d = Oct[['国内航线(大陆)']].values
Oct_i = Oct[['国际航线(加港澳台地区)']].values
Nov_amount = Nov[['旅客吞吐量(人次)']]
Nov_x = Nov_amount.values
Nov_d = Nov[['国内航线(大陆)']].values
Nov_i = Nov[['国际航线(加港澳台地区)']].values
Dec_amount = Dec[['旅客吞吐量(人次)']]
Dec_x = Dec_amount.values
Dec_d = Dec[['国内航线(大陆)']].values
Dec_i = Dec[['国际航线(加港澳台地区)']].values

plt.figure()
plt.title('各月变化')  # 添加标题
plt.xlabel('年份')  # x轴名称
plt.ylabel('客运量(万人)')  # y轴名称
x = np.arange(21)
x_labels = ['00','01','02','03','04','05','06','07','08','09','10','11','12'
            ,'13','14','15','16','17','18','19','20']
plt.xticks(x, x_labels)  # set labels
plt.plot(Jan_x)
plt.plot(Feb_x)
plt.plot(Mar_x)
plt.plot(Apr_x)
plt.plot(May_x)
plt.plot(Jun_x)
plt.plot(Jul_x)
plt.plot(Aug_x)
plt.plot(Sep_x)
plt.plot(Oct_x)
plt.plot(Nov_x)
plt.plot(Dec_x)
plt.legend(['Jan','Feb','Mar','Apr','May','Jun'
            ,'Jul','Aug','Sep','Oct','Nov','Dec'])  # set legend
plt.show()

plt.figure()
plt.title('January')  # 添加标题
plt.xlabel('年份')  # x轴名称
plt.ylabel('客运量(万人)')  # y轴名称
x = np.arange(21)
x_labels = ['00','01','02','03','04','05','06','07','08','09','10','11','12'
            ,'13','14','15','16','17','18','19','20']
plt.xticks(x, x_labels)  # set labels
plt.plot(Jan_x)
plt.plot(Jan_d)
plt.plot(Jan_i)
plt.legend(['Total','Mainland','International+HMT'])
plt.show()

plt.figure()
plt.title('February')  # 添加标题
plt.xlabel('年份')  # x轴名称
plt.ylabel('客运量(万人)')  # y轴名称
x = np.arange(21)
x_labels = ['00','01','02','03','04','05','06','07','08','09','10','11','12'
            ,'13','14','15','16','17','18','19','20']
plt.xticks(x, x_labels)  # set labels
plt.plot(Feb_x)
plt.plot(Feb_d)
plt.plot(Feb_i)
plt.legend(['Total','Mainland','International+HMT'])
plt.show()

plt.figure()
plt.title('March')  # 添加标题
plt.xlabel('年份')  # x轴名称
plt.ylabel('客运量(万人)')  # y轴名称
x = np.arange(21)
x_labels = ['00','01','02','03','04','05','06','07','08','09','10','11','12'
            ,'13','14','15','16','17','18','19','20']
plt.xticks(x, x_labels)  # set labels
plt.plot(Mar_x)
plt.plot(Mar_d)
plt.plot(Mar_i)
plt.legend(['Total','Mainland','International+HMT'])
plt.show()

plt.figure()
plt.title('April')  # 添加标题
plt.xlabel('年份')  # x轴名称
plt.ylabel('客运量(万人)')  # y轴名称
x = np.arange(21)
x_labels = ['00','01','02','03','04','05','06','07','08','09','10','11','12'
            ,'13','14','15','16','17','18','19','20']
plt.xticks(x, x_labels)  # set labels
plt.plot(Apr_x)
plt.plot(Apr_d)
plt.plot(Apr_i)
plt.legend(['Total','Mainland','International+HMT'])
plt.show()

plt.figure()
plt.title('May')  # 添加标题
plt.xlabel('年份')  # x轴名称
plt.ylabel('客运量(万人)')  # y轴名称
x = np.arange(21)
x_labels = ['00','01','02','03','04','05','06','07','08','09','10','11','12'
            ,'13','14','15','16','17','18','19','20']
plt.xticks(x, x_labels)  # set labels
plt.plot(May_x)
plt.plot(May_d)
plt.plot(May_i)
plt.legend(['Total','Mainland','International+HMT'])
plt.show()

plt.figure()
plt.title('June')  # 添加标题
plt.xlabel('年份')  # x轴名称
plt.ylabel('客运量(万人)')  # y轴名称
x = np.arange(21)
x_labels = ['00','01','02','03','04','05','06','07','08','09','10','11','12'
            ,'13','14','15','16','17','18','19','20']
plt.xticks(x, x_labels)  # set labels
plt.plot(Jun_x)
plt.plot(Jun_d)
plt.plot(Jun_i)
plt.legend(['Total','Mainland','International+HMT'])
plt.show()

plt.figure()
plt.title('July')  # 添加标题
plt.xlabel('年份')  # x轴名称
plt.ylabel('客运量(万人)')  # y轴名称
x = np.arange(21)
x_labels = ['00','01','02','03','04','05','06','07','08','09','10','11','12'
            ,'13','14','15','16','17','18','19','20']
plt.xticks(x, x_labels)  # set labels
plt.plot(Jul_x)
plt.plot(Jul_d)
plt.plot(Jul_i)
plt.legend(['Total','Mainland','International+HMT'])
plt.show()

plt.figure()
plt.title('August')  # 添加标题
plt.xlabel('年份')  # x轴名称
plt.ylabel('客运量(万人)')  # y轴名称
x = np.arange(21)
x_labels = ['00','01','02','03','04','05','06','07','08','09','10','11','12'
            ,'13','14','15','16','17','18','19','20']
plt.xticks(x, x_labels)  # set labels
plt.plot(Aug_x)
plt.plot(Aug_d)
plt.plot(Aug_i)
plt.legend(['Total','Mainland','International+HMT'])
plt.show()

plt.figure()
plt.title('September')  # 添加标题
plt.xlabel('年份')  # x轴名称
plt.ylabel('客运量(万人)')  # y轴名称
x = np.arange(21)
x_labels = ['00','01','02','03','04','05','06','07','08','09','10','11','12'
            ,'13','14','15','16','17','18','19','20']
plt.xticks(x, x_labels)  # set labels
plt.plot(Sep_x)
plt.plot(Sep_d)
plt.plot(Sep_i)
plt.legend(['Total','Mainland','International+HMT'])
plt.show()

plt.figure()
plt.title('October')  # 添加标题
plt.xlabel('年份')  # x轴名称
plt.ylabel('客运量(万人)')  # y轴名称
x = np.arange(21)
x_labels = ['00','01','02','03','04','05','06','07','08','09','10','11','12'
            ,'13','14','15','16','17','18','19','20']
plt.xticks(x, x_labels)  # set labels
plt.plot(Oct_x)
plt.plot(Oct_d)
plt.plot(Oct_i)
plt.legend(['Total','Mainland','International+HMT'])
plt.show()

plt.figure()
plt.title('November')  # 添加标题
plt.xlabel('年份')  # x轴名称
plt.ylabel('客运量(万人)')  # y轴名称
x = np.arange(21)
x_labels = ['00','01','02','03','04','05','06','07','08','09','10','11','12'
            ,'13','14','15','16','17','18','19','20']
plt.xticks(x, x_labels)  # set labels
plt.plot(Nov_x)
plt.plot(Nov_d)
plt.plot(Nov_i)
plt.legend(['Total','Mainland','International+HMT'])
plt.show()

plt.figure()
plt.title('December')  # 添加标题
plt.xlabel('年份')  # x轴名称
plt.ylabel('客运量(万人)')  # y轴名称
x = np.arange(21)
x_labels = ['00','01','02','03','04','05','06','07','08','09','10','11','12'
            ,'13','14','15','16','17','18','19','20']
plt.xticks(x, x_labels)  # set labels
plt.plot(Dec_x)
plt.plot(Dec_d)
plt.plot(Dec_i)
plt.legend(['Total','Mainland','International+HMT'])
plt.show()

# ====================================================================================


# ====================================== Rate of Change ================================

# =================================== total ==========================================
amount_change = df_rever[['旅客吞吐量同比增长']]
amount_change.plot()
plt.title('首都机场旅客吞吐量同比增长')
plt.xlabel('月-年')
plt.ylabel('旅客吞吐量同比增长(%)')
plt.savefig("首都机场旅客吞吐量同比增长.png")
# ====================================================================================

# ===================================== SARS =============================================
amount_change_sars = amount_change.loc['Sep-01':'Sep-06']
amount_change_sars.plot()
plt.title('首都机场旅客吞吐量同比增长(SARS)')
plt.xlabel('月-年')
plt.ylabel('旅客吞吐量同比增长(%)')
plt.savefig("首都机场旅客吞吐量同比增长.png")
# ====================================================================================

# ================================ Post-SARS + Regression =============    ==========================
amount_change_postsars = amount_change.loc['Sep-06':'Dec-19']
x_axis = np.arange(amount_change_postsars.size) + 1
Xtrain, Xvalid, ytrain, yvalid = train_test_split(x_axis,amount_change_postsars,random_state=0,test_size=32)
ytrain, yvalid = ytrain.values[:,0], yvalid.values[:,0]
Xtrain = Xtrain.reshape((128,1))
Xvalid = Xvalid.reshape((32,1))

# # linear regression:
# # model = linear_model.Ridge(alpha=0.01)
# # model = linear_model.SGDRegressor(loss='huber', penalty='l1', alpha=0.0001)
# model = linear_model.LinearRegression()
# # model = linear_model.Lasso()
# model.fit(Xtrain, ytrain)
# ytrain_pred = model.predict(Xtrain)
# training_error = np.average((ytrain_pred - np.array(ytrain)) ** 2)
# yvalid_pred = model.predict(Xvalid)
# validation_error = np.average((yvalid_pred - np.array(yvalid)) ** 2)
# plt.figure()
# plt.title('首都机场旅客吞吐量同比增长(post-SARS)')
# plt.xlabel('月-年')
# plt.ylabel('旅客吞吐量同比增长(%)')
# plt.plot(amount_change_postsars.values)
# plt.plot(Xtrain, ytrain_pred)
# plt.show()
# print(training_error / 10**12, validation_error / 10**12)
# # plt.savefig("首都机场旅客吞吐量同比增长.png")

# # polynomial regression degree-2:
# quadratic_featurizer = PolynomialFeatures(degree=2)
# X_train_quadratic = quadratic_featurizer.fit_transform(Xtrain)
# X_valid_quadratic = quadratic_featurizer.fit_transform(Xvalid)
# model = linear_model.Lasso()
# model.fit(X_train_quadratic, ytrain)
# xx = np.linspace(0, 160, 160)
# xx_quadratic = quadratic_featurizer.transform(xx.reshape(xx.shape[0], 1))
# yy_predict = model.predict(xx_quadratic)
# ytrain_pred = model.predict(X_train_quadratic)
# training_error = np.average((ytrain_pred - np.array(ytrain)) ** 2)
# yvalid_pred = model.predict(X_valid_quadratic)
# validation_error = np.average((yvalid_pred - np.array(yvalid)) ** 2)
# plt.figure()
# plt.title('首都机场旅客吞吐量同比增长(post-SARS)')
# plt.xlabel('月-年')
# plt.ylabel('旅客吞吐量同比增长(%)')
# plt.plot(amount_change_postsars.values)
# plt.plot(xx, yy_predict)
# plt.show()
# print(training_error / 10**12, validation_error / 10**12)

# polynomial regression degree-3:
quadratic_featurizer = PolynomialFeatures(degree=3)
X_train_quadratic = quadratic_featurizer.fit_transform(Xtrain)
X_valid_quadratic = quadratic_featurizer.fit_transform(Xvalid)
# model = linear_model.Lasso(max_iter=10000)
model = linear_model.LinearRegression()
model.fit(X_train_quadratic, ytrain)
xx = np.linspace(0, 160, 160)
xx_quadratic = quadratic_featurizer.transform(xx.reshape(xx.shape[0], 1))
yy_predict = model.predict(xx_quadratic)
ytrain_pred = model.predict(X_train_quadratic)
training_error = np.average((ytrain_pred - np.array(ytrain)) ** 2)
yvalid_pred = model.predict(X_valid_quadratic)
validation_error = np.average((yvalid_pred - np.array(yvalid)) ** 2)
plt.figure()
plt.title('首都机场旅客吞吐量同比增长(post-SARS)')
plt.xlabel('月-年')
plt.ylabel('旅客吞吐量同比增长(%)')
plt.plot(amount_change_postsars.values)
plt.plot(xx, yy_predict)
plt.show()
print(training_error / 10**12, validation_error / 10**12)

# # polynomial regression degree-4:
# quadratic_featurizer = PolynomialFeatures(degree=4)
# X_train_quadratic = quadratic_featurizer.fit_transform(Xtrain)
# X_valid_quadratic = quadratic_featurizer.fit_transform(Xvalid)
# # model = linear_model.Lasso(max_iter=10000)
# model = linear_model.LinearRegression()
# model.fit(X_train_quadratic, ytrain)
# xx = np.linspace(0, 160, 160)
# xx_quadratic = quadratic_featurizer.transform(xx.reshape(xx.shape[0], 1))
# yy_predict = model.predict(xx_quadratic)
# ytrain_pred = model.predict(X_train_quadratic)
# training_error = np.average((ytrain_pred - np.array(ytrain)) ** 2)
# yvalid_pred = model.predict(X_valid_quadratic)
# validation_error = np.average((yvalid_pred - np.array(yvalid)) ** 2)
# plt.figure()
# plt.title('首都机场旅客吞吐量同比增长(post-SARS)')
# plt.xlabel('月-年')
# plt.ylabel('旅客吞吐量同比增长(%)')
# plt.plot(amount_change_postsars.values)
# plt.plot(xx, yy_predict)
# plt.show()
# print(training_error / 10**12, validation_error / 10**12)
# ====================================================================================

# ====================================== COVID =======================================
amount_change_covid = amount_change.loc['Jun-19':]
amount_change_covid.plot()
plt.title('首都机场旅客吞吐量同比增长(COVID)')
plt.xlabel('月-年')
plt.ylabel('旅客吞吐量同比增长(%)')
plt.savefig("首都机场旅客吞吐量同比增长.png")

# ====================================================================================

 
# ======================================================================================
# train = amount.loc['Jan-00':'Jan-16']
# test = amount.loc['Jan-16':'Dec-19']

# y_hat_avg = test.copy()
# fit1 = ExponentialSmoothing(np.asarray(train.values), seasonal_periods=12, trend='add', seasonal='add', ).fit()

# y_hat_avg['Holt_Winter'] = fit1.forecast(len(test))
# # plt.figure(figsize=(16, 8))
# plt.figure()
# plt.plot(train, label='Train')
# plt.plot(test, label='Test')
# plt.plot(y_hat_avg['Holt_Winter'], label='Holt_Winter')
# plt.legend(loc='best')
# plt.show()

# error = np.average((y_hat_avg['Holt_Winter'] - y_hat_avg['旅客吞吐量(人次)']) ** 2)
# print(error / 10**12)
# ======================================================================================


# ======================================================================================
train = amount.loc['Jan-00':'Jan-16']
test = amount.loc['Jan-16':'Dec-19']
    
y_hat_avg = test.copy()
model = ExponentialSmoothing(np.asarray(train.values), seasonal_periods=24, trend='add', seasonal='add', ).fit()
    
y_hat_avg['Holt_Winter'] = model.forecast(len(test))
plt.figure()
plt.plot(train, label='Train')
plt.plot(test, label='Test')
plt.plot(y_hat_avg['Holt_Winter'], label='Holt_Winter')
plt.legend(loc='best')
plt.show()
    
error = np.average((y_hat_avg['Holt_Winter'] - y_hat_avg['旅客吞吐量(人次)']) ** 2) / 10**12
print(error)
# =====================================================================================



# ======================================================================================
pre_COVID = amount.loc[:'Dec-19']

post_COVID = amount.loc['Dec-19':]

model = ExponentialSmoothing(np.asarray(pre_COVID.values), seasonal_periods=24, trend='add', seasonal='add', ).fit()
    
# y_hat_avg['Holt_Winter'] = model.forecast(len(test))
predict = model.forecast(19)

# for i in range(18):
#     post_COVID[i+7] = predict[i]

post_COVID = post_COVID.copy()
post_COVID.loc['Jan-20'] = predict[0]
post_COVID.loc['Feb-20',:] = predict[1]
post_COVID.loc['Mar-20',:] = predict[2]
post_COVID.loc['Apr-20',:] = predict[3]
post_COVID.loc['May-20',:] = predict[4]
post_COVID.loc['Jun-20',:] = predict[5]
post_COVID.loc['Jul-20',:] = predict[6]
post_COVID.loc['Aug-20',:] = predict[7]
post_COVID.loc['Sep-20',:] = predict[8]
post_COVID.loc['Oct-20',:] = predict[9]
post_COVID.loc['Nov-20',:] = predict[10]
post_COVID.loc['Dec-20',:] = predict[11]
post_COVID.loc['Jan-21',:] = predict[12]
post_COVID.loc['Feb-21',:] = predict[13]
post_COVID.loc['Mar-21',:] = predict[14]
post_COVID.loc['Apr-21',:] = predict[15]
post_COVID.loc['May-21',:] = predict[16]
post_COVID.loc['Jun-21',:] = predict[17]
post_COVID.loc['Jul-21',:] = predict[18]

plt.figure()
# plt.xticks(rotation=300)
# plt.plot(train, label='Train')
plt.plot(pre_COVID, label='pre_COVID')
# plt.plot(y_hat_avg['Holt_Winter'], label='Holt_Winter')
plt.plot(post_COVID, label='assumed_without_COVID')
plt.legend(loc='best')
plt.show()
    
error = np.average((y_hat_avg['Holt_Winter'] - y_hat_avg['旅客吞吐量(人次)']) ** 2) / 10**12
print(error)
# pre_COVID = amount.loc[:'Dec-19']
# post_COVID = model.forecast(18)
# plt.figure()
# plt.plot(pre_COVID, label='pre-COVID')
# plt.plot(post_COVID, label='post_COVID')
# plt.plot(predict, label='The coming 12 months')
# plt.legend(loc='best')
# plt.show()
# ======================================================================================


df_week = pd.read_csv("~/Desktop/普华永道/首都机场周变化量.csv",header = 0,index_col=0)
df_week_t=df_week.T

# domestic_week = df_week_t[['国内航线旅客量同比变化(%)']]



# model = ExponentialSmoothing(np.asarray(domestic_week.values), seasonal_periods=5, trend='add', seasonal='add', ).fit()
# predict = model.forecast(13)



# domestic_week_copy = domestic_week.copy()
# domestic_week_copy.loc['Jul_Week4',:] = predict[0]
# domestic_week_copy.loc['Aug_Week1',:] = predict[1]
# domestic_week_copy.loc['Aug_Week2',:] = predict[2]
# domestic_week_copy.loc['Aug_Week3',:] = predict[3]
# domestic_week_copy.loc['Aug_Week4',:] = predict[4]
# domestic_week_copy.loc['Sep_Week1',:] = predict[5]
# domestic_week_copy.loc['Sep_Week2',:] = predict[6]
# domestic_week_copy.loc['Sep_Week3',:] = predict[7]
# domestic_week_copy.loc['Sep_Week4',:] = predict[8]
# domestic_week_copy.loc['Oct_Week1',:] = predict[9]
# domestic_week_copy.loc['Oct_Week2',:] = predict[10]
# domestic_week_copy.loc['Oct_Week3',:] = predict[11]
# domestic_week_copy.loc['Oct_Week4',:] = predict[12]



# # x = np.arange(29).reshape((29,1))
# # polynomial regression degree-4:
# # quadratic_featurizer = PolynomialFeatures(degree=4)
# # X_train_quadratic = quadratic_featurizer.fit_transform(x)
# # model = linear_model.Lasso(max_iter=100000)
# # model = linear_model.LinearRegression()
# # model.fit(X_train_quadratic, domestic_week.values)
# # xx = np.linspace(0, 42, 42)
# # xx_quadratic = quadratic_featurizer.transform(xx.reshape(xx.shape[0], 1))
# # yy_predict = model.predict(xx_quadratic)
# plt.figure()
# plt.title('国内航线旅客量同比变化')
# plt.xlabel('月-年')
# x = np.arange(42)
# x_labels = ['Jan_Week1','Jan_Week2','Jan_Week3','Jan_Week4','Jan_Week5',
#             'Feb_Week1','Feb_Week2','Feb_Week3','Feb_Week4',
#             'Mar_Week1','Mar_Week2','Mar_Week3','Mar_Week4',
#             'Apr_Week1','Apr_Week2','Apr_Week3','Apr_Week4','Apr_Week5',
#             'May_Week1','May_Week2','May_Week3','May_Week4',
#             'Jun_Week1','Jun_Week2','Jun_Week3','Jun_Week4',
#             'Jul_Week1','Jul_Week2','Jul_Week3','Jul_Week4',
#             'Aug_Week1','Aug_Week2','Aug_Week3','Aug_Week4',
#             'Sep_Week1','Sep_Week2','Sep_Week3','Sep_Week4',
#             'Oct_Week1','Oct_Week2','Oct_Week3','Oct_Week4']
# plt.xticks(x, x_labels,rotation=270)
# plt.ylabel('旅客量同比增长(%)')
# plt.plot(domestic_week_copy.values)
# # plt.plot(yy_predict)
# plt.show()





domestic_week = df_week_t[['国内航线旅客量同比变化(%)']]
plt.figure()
plt.title('国内航线旅客量同比变化')
plt.xlabel('月-年')
x = np.arange(29)
x_labels = ['Jan_Week1','Jan_Week2','Jan_Week3','Jan_Week4','Jan_Week5',
            'Feb_Week1','Feb_Week2','Feb_Week3','Feb_Week4',
            'Mar_Week1','Mar_Week2','Mar_Week3','Mar_Week4',
            'Apr_Week1','Apr_Week2','Apr_Week3','Apr_Week4','Apr_Week5',
            'May_Week1','May_Week2','May_Week3','May_Week4',
            'Jun_Week1','Jun_Week2','Jun_Week3','Jun_Week4',
            'Jul_Week1','Jul_Week2','Jul_Week3']
plt.xticks(x, x_labels,rotation=270)
plt.ylabel('旅客量同比增长(%)')
plt.plot(domestic_week.values)
plt.show()






df_predict = pd.read_csv("~/Desktop/普华永道/首都机场周变化量预测.csv",header = 0,index_col=0)
df_predict=df_predict.T

predict_1 = df_predict[['无反弹国内航线旅客量同比变化(%)']]
plt.figure(figsize=(16,8))
plt.title('国内航线旅客量同比变化')
plt.xlabel('月-年')
x = np.arange(78)
x_labels = ['Jan_Week1','Jan_Week2','Jan_Week3','Jan_Week4','Jan_Week5',
            'Feb_Week1','Feb_Week2','Feb_Week3','Feb_Week4',
            'Mar_Week1','Mar_Week2','Mar_Week3','Mar_Week4',
            'Apr_Week1','Apr_Week2','Apr_Week3','Apr_Week4','Apr_Week5',
            'May_Week1','May_Week2','May_Week3','May_Week4',
            'Jun_Week1','Jun_Week2','Jun_Week3','Jun_Week4',
            'Jul_Week1','Jul_Week2','Jul_Week3','Jul_Week4',
            'Aug_Week1','Aug_Week2','Aug_Week3','Aug_Week4',
            'Sep_Week1','Sep_Week2','Sep_Week3','Sep_Week4',
            'Oct_Week1','Oct_Week2','Oct_Week3','Oct_Week4',
            'Nov_Week1','Nov_Week2','Nov_Week3','Nov_Week4',
            'Dec_Week1','Dec_Week2','Dec_Week3','Dec_Week4',
            'Jan_Week1','Jan_Week2','Jan_Week3','Jan_Week4',
            'Feb_Week1','Feb_Week2','Feb_Week3','Feb_Week4',
            'Mar_Week1','Mar_Week2','Mar_Week3','Mar_Week4',
            'Apr_Week1','Apr_Week2','Apr_Week3','Apr_Week4',
            'May_Week1','May_Week2','May_Week3','May_Week4',
            'Jun_Week1','Jun_Week2','Jun_Week3','Jun_Week4',
            'Jul_Week1','Jul_Week2','Jul_Week3','Jul_Week4']
plt.xticks(x, x_labels,rotation=270)
plt.ylabel('旅客量同比增长(%)')
plt.plot(predict_1.values)
plt.show()


predict_2 = df_predict[['暑期反弹国内航线旅客量同比变化(%)']]
plt.figure(figsize=(16,8))
plt.title('国内航线旅客量同比变化')
plt.xlabel('月-年')
x = np.arange(78)
x_labels = ['Jan_Week1','Jan_Week2','Jan_Week3','Jan_Week4','Jan_Week5',
            'Feb_Week1','Feb_Week2','Feb_Week3','Feb_Week4',
            'Mar_Week1','Mar_Week2','Mar_Week3','Mar_Week4',
            'Apr_Week1','Apr_Week2','Apr_Week3','Apr_Week4','Apr_Week5',
            'May_Week1','May_Week2','May_Week3','May_Week4',
            'Jun_Week1','Jun_Week2','Jun_Week3','Jun_Week4',
            'Jul_Week1','Jul_Week2','Jul_Week3','Jul_Week4',
            'Aug_Week1','Aug_Week2','Aug_Week3','Aug_Week4',
            'Sep_Week1','Sep_Week2','Sep_Week3','Sep_Week4',
            'Oct_Week1','Oct_Week2','Oct_Week3','Oct_Week4',
            'Nov_Week1','Nov_Week2','Nov_Week3','Nov_Week4',
            'Dec_Week1','Dec_Week2','Dec_Week3','Dec_Week4',
            'Jan_Week1','Jan_Week2','Jan_Week3','Jan_Week4',
            'Feb_Week1','Feb_Week2','Feb_Week3','Feb_Week4',
            'Mar_Week1','Mar_Week2','Mar_Week3','Mar_Week4',
            'Apr_Week1','Apr_Week2','Apr_Week3','Apr_Week4',
            'May_Week1','May_Week2','May_Week3','May_Week4',
            'Jun_Week1','Jun_Week2','Jun_Week3','Jun_Week4',
            'Jul_Week1','Jul_Week2','Jul_Week3','Jul_Week4']
plt.xticks(x, x_labels,rotation=270)
plt.ylabel('旅客量同比增长(%)')
plt.plot(predict_2.values)
plt.show()


predict_3 = df_predict[['冬季反弹国内航线旅客量同比变化(%)']]
plt.figure(figsize=(16,8))
plt.title('国内航线旅客量同比变化')
plt.xlabel('月-年')
x = np.arange(78)
x_labels = ['Jan_Week1','Jan_Week2','Jan_Week3','Jan_Week4','Jan_Week5',
            'Feb_Week1','Feb_Week2','Feb_Week3','Feb_Week4',
            'Mar_Week1','Mar_Week2','Mar_Week3','Mar_Week4',
            'Apr_Week1','Apr_Week2','Apr_Week3','Apr_Week4','Apr_Week5',
            'May_Week1','May_Week2','May_Week3','May_Week4',
            'Jun_Week1','Jun_Week2','Jun_Week3','Jun_Week4',
            'Jul_Week1','Jul_Week2','Jul_Week3','Jul_Week4',
            'Aug_Week1','Aug_Week2','Aug_Week3','Aug_Week4',
            'Sep_Week1','Sep_Week2','Sep_Week3','Sep_Week4',
            'Oct_Week1','Oct_Week2','Oct_Week3','Oct_Week4',
            'Nov_Week1','Nov_Week2','Nov_Week3','Nov_Week4',
            'Dec_Week1','Dec_Week2','Dec_Week3','Dec_Week4',
            'Jan_Week1','Jan_Week2','Jan_Week3','Jan_Week4',
            'Feb_Week1','Feb_Week2','Feb_Week3','Feb_Week4',
            'Mar_Week1','Mar_Week2','Mar_Week3','Mar_Week4',
            'Apr_Week1','Apr_Week2','Apr_Week3','Apr_Week4',
            'May_Week1','May_Week2','May_Week3','May_Week4',
            'Jun_Week1','Jun_Week2','Jun_Week3','Jun_Week4',
            'Jul_Week1','Jul_Week2','Jul_Week3','Jul_Week4']
plt.xticks(x, x_labels,rotation=270)
plt.ylabel('旅客量同比增长(%)')
plt.plot(predict_3.values)
plt.show()


# ==================================================================================

pre_COVID_domestic = domestic.loc[:'Dec-19']

post_COVID_domestic = domestic.loc['Dec-19':]

model = ExponentialSmoothing(np.asarray(pre_COVID_domestic.values), seasonal_periods=24, trend='add', seasonal='add', ).fit()
    
# y_hat_avg['Holt_Winter'] = model.forecast(len(test))
predict = model.forecast(19)

post_COVID_domestic = post_COVID_domestic.copy()
post_COVID_domestic.loc['Jan-20'] = predict[0]
post_COVID_domestic.loc['Feb-20',:] = predict[1]
post_COVID_domestic.loc['Mar-20',:] = predict[2]
post_COVID_domestic.loc['Apr-20',:] = predict[3]
post_COVID_domestic.loc['May-20',:] = predict[4]
post_COVID_domestic.loc['Jun-20',:] = predict[5]
post_COVID_domestic.loc['Jul-20',:] = predict[6]
post_COVID_domestic.loc['Aug-20',:] = predict[7]
post_COVID_domestic.loc['Sep-20',:] = predict[8]
post_COVID_domestic.loc['Oct-20',:] = predict[9]
post_COVID_domestic.loc['Nov-20',:] = predict[10]
post_COVID_domestic.loc['Dec-20',:] = predict[11]
post_COVID_domestic.loc['Jan-21',:] = predict[12]
post_COVID_domestic.loc['Feb-21',:] = predict[13]
post_COVID_domestic.loc['Mar-21',:] = predict[14]
post_COVID_domestic.loc['Apr-21',:] = predict[15]
post_COVID_domestic.loc['May-21',:] = predict[16]
post_COVID_domestic.loc['Jun-21',:] = predict[17]
post_COVID_domestic.loc['Jul-21',:] = predict[18]


pre_COVID_foreign = foreign.loc[:'Dec-19']

post_COVID_foreign = foreign.loc['Dec-19':]

model = ExponentialSmoothing(np.asarray(pre_COVID_foreign.values), seasonal_periods=24, trend='add', seasonal='add', ).fit()
    
# y_hat_avg['Holt_Winter'] = model.forecast(len(test))
predict = model.forecast(19)

post_COVID_foreign = post_COVID_foreign.copy()
post_COVID_foreign.loc['Jan-20'] = predict[0]
post_COVID_foreign.loc['Feb-20',:] = predict[1]
post_COVID_foreign.loc['Mar-20',:] = predict[2]
post_COVID_foreign.loc['Apr-20',:] = predict[3]
post_COVID_foreign.loc['May-20',:] = predict[4]
post_COVID_foreign.loc['Jun-20',:] = predict[5]
post_COVID_foreign.loc['Jul-20',:] = predict[6]
post_COVID_foreign.loc['Aug-20',:] = predict[7]
post_COVID_foreign.loc['Sep-20',:] = predict[8]
post_COVID_foreign.loc['Oct-20',:] = predict[9]
post_COVID_foreign.loc['Nov-20',:] = predict[10]
post_COVID_foreign.loc['Dec-20',:] = predict[11]
post_COVID_foreign.loc['Jan-21',:] = predict[12]
post_COVID_foreign.loc['Feb-21',:] = predict[13]
post_COVID_foreign.loc['Mar-21',:] = predict[14]
post_COVID_foreign.loc['Apr-21',:] = predict[15]
post_COVID_foreign.loc['May-21',:] = predict[16]
post_COVID_foreign.loc['Jun-21',:] = predict[17]
post_COVID_foreign.loc['Jul-21',:] = predict[18]




foreign_week = df_week_t[['国际航线旅客量同比变化(%)']]
plt.figure()
plt.title('国际航线旅客量同比变化')
plt.xlabel('月-年')
x = np.arange(29)
x_labels = ['Jan_Week1','Jan_Week2','Jan_Week3','Jan_Week4','Jan_Week5',
            'Feb_Week1','Feb_Week2','Feb_Week3','Feb_Week4',
            'Mar_Week1','Mar_Week2','Mar_Week3','Mar_Week4',
            'Apr_Week1','Apr_Week2','Apr_Week3','Apr_Week4','Apr_Week5',
            'May_Week1','May_Week2','May_Week3','May_Week4',
            'Jun_Week1','Jun_Week2','Jun_Week3','Jun_Week4',
            'Jul_Week1','Jul_Week2','Jul_Week3']
plt.xticks(x, x_labels,rotation=270)
plt.ylabel('旅客量同比增长(%)')
plt.plot(foreign_week.values)
plt.show()



predict_4 = df_predict[['无反弹国际航线旅客量同比变化(%)']]
plt.figure(figsize=(16,8))
plt.title('国际航线旅客量同比变化')
plt.xlabel('月-年')
x = np.arange(78)
x_labels = ['Jan_Week1','Jan_Week2','Jan_Week3','Jan_Week4','Jan_Week5',
            'Feb_Week1','Feb_Week2','Feb_Week3','Feb_Week4',
            'Mar_Week1','Mar_Week2','Mar_Week3','Mar_Week4',
            'Apr_Week1','Apr_Week2','Apr_Week3','Apr_Week4','Apr_Week5',
            'May_Week1','May_Week2','May_Week3','May_Week4',
            'Jun_Week1','Jun_Week2','Jun_Week3','Jun_Week4',
            'Jul_Week1','Jul_Week2','Jul_Week3','Jul_Week4',
            'Aug_Week1','Aug_Week2','Aug_Week3','Aug_Week4',
            'Sep_Week1','Sep_Week2','Sep_Week3','Sep_Week4',
            'Oct_Week1','Oct_Week2','Oct_Week3','Oct_Week4',
            'Nov_Week1','Nov_Week2','Nov_Week3','Nov_Week4',
            'Dec_Week1','Dec_Week2','Dec_Week3','Dec_Week4',
            'Jan_Week1','Jan_Week2','Jan_Week3','Jan_Week4',
            'Feb_Week1','Feb_Week2','Feb_Week3','Feb_Week4',
            'Mar_Week1','Mar_Week2','Mar_Week3','Mar_Week4',
            'Apr_Week1','Apr_Week2','Apr_Week3','Apr_Week4',
            'May_Week1','May_Week2','May_Week3','May_Week4',
            'Jun_Week1','Jun_Week2','Jun_Week3','Jun_Week4',
            'Jul_Week1','Jul_Week2','Jul_Week3','Jul_Week4']
plt.xticks(x, x_labels,rotation=270)
plt.ylabel('旅客量同比增长(%)')
plt.plot(predict_4.values)
plt.show()


predict_5 = df_predict[['暑期反弹国际航线旅客量同比变化(%)']]
plt.figure(figsize=(16,8))
plt.title('国际航线旅客量同比变化')
plt.xlabel('月-年')
x = np.arange(78)
x_labels = ['Jan_Week1','Jan_Week2','Jan_Week3','Jan_Week4','Jan_Week5',
            'Feb_Week1','Feb_Week2','Feb_Week3','Feb_Week4',
            'Mar_Week1','Mar_Week2','Mar_Week3','Mar_Week4',
            'Apr_Week1','Apr_Week2','Apr_Week3','Apr_Week4','Apr_Week5',
            'May_Week1','May_Week2','May_Week3','May_Week4',
            'Jun_Week1','Jun_Week2','Jun_Week3','Jun_Week4',
            'Jul_Week1','Jul_Week2','Jul_Week3','Jul_Week4',
            'Aug_Week1','Aug_Week2','Aug_Week3','Aug_Week4',
            'Sep_Week1','Sep_Week2','Sep_Week3','Sep_Week4',
            'Oct_Week1','Oct_Week2','Oct_Week3','Oct_Week4',
            'Nov_Week1','Nov_Week2','Nov_Week3','Nov_Week4',
            'Dec_Week1','Dec_Week2','Dec_Week3','Dec_Week4',
            'Jan_Week1','Jan_Week2','Jan_Week3','Jan_Week4',
            'Feb_Week1','Feb_Week2','Feb_Week3','Feb_Week4',
            'Mar_Week1','Mar_Week2','Mar_Week3','Mar_Week4',
            'Apr_Week1','Apr_Week2','Apr_Week3','Apr_Week4',
            'May_Week1','May_Week2','May_Week3','May_Week4',
            'Jun_Week1','Jun_Week2','Jun_Week3','Jun_Week4',
            'Jul_Week1','Jul_Week2','Jul_Week3','Jul_Week4']
plt.xticks(x, x_labels,rotation=270)
plt.ylabel('旅客量同比增长(%)')
plt.plot(predict_5.values)
plt.show()


predict_6 = df_predict[['冬季反弹国际航线旅客量同比变化(%)']]
plt.figure(figsize=(16,8))
plt.title('国际航线旅客量同比变化')
plt.xlabel('月-年')
x = np.arange(78)
x_labels = ['Jan_Week1','Jan_Week2','Jan_Week3','Jan_Week4','Jan_Week5',
            'Feb_Week1','Feb_Week2','Feb_Week3','Feb_Week4',
            'Mar_Week1','Mar_Week2','Mar_Week3','Mar_Week4',
            'Apr_Week1','Apr_Week2','Apr_Week3','Apr_Week4','Apr_Week5',
            'May_Week1','May_Week2','May_Week3','May_Week4',
            'Jun_Week1','Jun_Week2','Jun_Week3','Jun_Week4',
            'Jul_Week1','Jul_Week2','Jul_Week3','Jul_Week4',
            'Aug_Week1','Aug_Week2','Aug_Week3','Aug_Week4',
            'Sep_Week1','Sep_Week2','Sep_Week3','Sep_Week4',
            'Oct_Week1','Oct_Week2','Oct_Week3','Oct_Week4',
            'Nov_Week1','Nov_Week2','Nov_Week3','Nov_Week4',
            'Dec_Week1','Dec_Week2','Dec_Week3','Dec_Week4',
            'Jan_Week1','Jan_Week2','Jan_Week3','Jan_Week4',
            'Feb_Week1','Feb_Week2','Feb_Week3','Feb_Week4',
            'Mar_Week1','Mar_Week2','Mar_Week3','Mar_Week4',
            'Apr_Week1','Apr_Week2','Apr_Week3','Apr_Week4',
            'May_Week1','May_Week2','May_Week3','May_Week4',
            'Jun_Week1','Jun_Week2','Jun_Week3','Jun_Week4',
            'Jul_Week1','Jul_Week2','Jul_Week3','Jul_Week4']
plt.xticks(x, x_labels,rotation=270)
plt.ylabel('旅客量同比增长(%)')
plt.plot(predict_6.values)
plt.show()


# =================================== 无反弹 ================================================
change_rate = np.arange(19)
change_rate[0] = (predict_1.values[0]+predict_1.values[1]+predict_1.values[2]+predict_1.values[3]+predict_1.values[4])/5
change_rate[1] = (predict_1.values[5]+predict_1.values[6]+predict_1.values[7]+predict_1.values[8])/4
change_rate[2] = (predict_1.values[9]+predict_1.values[10]+predict_1.values[11]+predict_1.values[12])/4
change_rate[3] =  (predict_1.values[13]+predict_1.values[14]+predict_1.values[15]+predict_1.values[16]+predict_1.values[17])/5
change_rate[4] = (predict_1.values[18]+predict_1.values[19]+predict_1.values[20]+predict_1.values[21])/4
change_rate[5] = (predict_1.values[22]+predict_1.values[23]+predict_1.values[24]+predict_1.values[25])/4
change_rate[6] = (predict_1.values[26]+predict_1.values[27]+predict_1.values[28]+predict_1.values[29])/4
change_rate[7] = (predict_1.values[30]+predict_1.values[31]+predict_1.values[32]+predict_1.values[33])/4
change_rate[8] = (predict_1.values[34]+predict_1.values[35]+predict_1.values[36]+predict_1.values[37])/4
change_rate[9] = (predict_1.values[38]+predict_1.values[39]+predict_1.values[40]+predict_1.values[41])/4
change_rate[10] = (predict_1.values[42]+predict_1.values[43]+predict_1.values[44]+predict_1.values[45])/4
change_rate[11] = (predict_1.values[46]+predict_1.values[47]+predict_1.values[48]+predict_1.values[49])/4
change_rate[12] = (predict_1.values[50]+predict_1.values[51]+predict_1.values[52]+predict_1.values[53])/4
change_rate[13] = (predict_1.values[54]+predict_1.values[55]+predict_1.values[56]+predict_1.values[57])/4
change_rate[14] = (predict_1.values[58]+predict_1.values[59]+predict_1.values[60]+predict_1.values[61])/4
change_rate[15] = (predict_1.values[62]+predict_1.values[63]+predict_1.values[64]+predict_1.values[65])/4
change_rate[16] = (predict_1.values[66]+predict_1.values[67]+predict_1.values[68]+predict_1.values[69])/4
change_rate[17] = (predict_1.values[70]+predict_1.values[71]+predict_1.values[72]+predict_1.values[73])/4
change_rate[18] = (predict_1.values[74]+predict_1.values[75]+predict_1.values[76]+predict_1.values[77])/4

post_COVID = domestic.loc['Dec-19':]


predict_val = (post_COVID_domestic.values[1,:].T * (change_rate.T / 100)) + post_COVID_domestic.values[1,:].T

predict_amount = post_COVID.copy()
predict_amount.loc['Jun-20',:] = domestic.loc['Jun-20']
predict_amount.loc['Jul-20',:] = predict_val[6]
predict_amount.loc['Aug-20',:] = predict_val[7]
predict_amount.loc['Sep-20',:] = predict_val[8]
predict_amount.loc['Oct-20',:] = predict_val[9]
predict_amount.loc['Nov-20',:] = predict_val[10]
predict_amount.loc['Dec-20',:] = predict_val[11]
predict_amount.loc['Jan-21',:] = predict_val[12]
predict_amount.loc['Feb-21',:] = predict_val[13]
predict_amount.loc['Mar-21',:] = predict_val[14]
predict_amount.loc['Apr-21',:] = predict_val[15]
predict_amount.loc['May-21',:] = predict_val[16]
predict_amount.loc['Jun-21',:] = predict_val[17]
predict_amount.loc['Jul-21',:] = predict_val[18]
predict_domestic = predict_amount.loc['Jun-20':]
predict_domestic = predict_domestic.copy()
predict_domestic.rename(columns={'国内航线(大陆)':'总旅客量'},inplace=True)


change_rate = np.arange(19)
change_rate[0] = (predict_4.values[0]+predict_4.values[1]+predict_4.values[2]+predict_4.values[3]+predict_4.values[4])/5
change_rate[1] = (predict_4.values[5]+predict_4.values[6]+predict_4.values[7]+predict_4.values[8])/4
change_rate[2] = (predict_4.values[9]+predict_4.values[10]+predict_4.values[11]+predict_4.values[12])/4
change_rate[3] =  (predict_4.values[13]+predict_4.values[14]+predict_4.values[15]+predict_4.values[16]+predict_4.values[17])/5
change_rate[4] = (predict_4.values[18]+predict_4.values[19]+predict_4.values[20]+predict_4.values[21])/4
change_rate[5] = (predict_4.values[22]+predict_4.values[23]+predict_4.values[24]+predict_4.values[25])/4
change_rate[6] = (predict_4.values[26]+predict_4.values[27]+predict_4.values[28]+predict_4.values[29])/4
change_rate[7] = (predict_4.values[30]+predict_4.values[31]+predict_4.values[32]+predict_4.values[33])/4
change_rate[8] = (predict_4.values[34]+predict_4.values[35]+predict_4.values[36]+predict_4.values[37])/4
change_rate[9] = (predict_4.values[38]+predict_4.values[39]+predict_4.values[40]+predict_4.values[41])/4
change_rate[10] = (predict_4.values[42]+predict_4.values[43]+predict_4.values[44]+predict_4.values[45])/4
change_rate[11] = (predict_4.values[46]+predict_4.values[47]+predict_4.values[48]+predict_4.values[49])/4
change_rate[12] = (predict_4.values[50]+predict_4.values[51]+predict_4.values[52]+predict_4.values[53])/4
change_rate[13] = (predict_4.values[54]+predict_4.values[55]+predict_4.values[56]+predict_4.values[57])/4
change_rate[14] = (predict_4.values[58]+predict_4.values[59]+predict_4.values[60]+predict_4.values[61])/4
change_rate[15] = (predict_4.values[62]+predict_4.values[63]+predict_4.values[64]+predict_4.values[65])/4
change_rate[16] = (predict_4.values[66]+predict_4.values[67]+predict_4.values[68]+predict_4.values[69])/4
change_rate[17] = (predict_4.values[70]+predict_4.values[71]+predict_4.values[72]+predict_4.values[73])/4
change_rate[18] = (predict_4.values[74]+predict_4.values[75]+predict_4.values[76]+predict_4.values[77])/4

post_COVID = foreign.loc['Dec-19':]

predict_val = (post_COVID_foreign.values[1,:].T * (change_rate.T / 100)) + post_COVID_foreign.values[1,:].T

predict_amount = post_COVID.copy()
predict_amount.loc['Jun-20',:] = foreign.loc['Jun-20']
predict_amount.loc['Jul-20',:] = predict_val[6]
predict_amount.loc['Aug-20',:] = predict_val[7]
predict_amount.loc['Sep-20',:] = predict_val[8]
predict_amount.loc['Oct-20',:] = predict_val[9]
predict_amount.loc['Nov-20',:] = predict_val[10]
predict_amount.loc['Dec-20',:] = predict_val[11]
predict_amount.loc['Jan-21',:] = predict_val[12]
predict_amount.loc['Feb-21',:] = predict_val[13]
predict_amount.loc['Mar-21',:] = predict_val[14]
predict_amount.loc['Apr-21',:] = predict_val[15]
predict_amount.loc['May-21',:] = predict_val[16]
predict_amount.loc['Jun-21',:] = predict_val[17]
predict_amount.loc['Jul-21',:] = predict_val[18]
predict_foreign = predict_amount.loc['Jun-20':]
predict_foreign = predict_foreign.copy()
predict_foreign.rename(columns={'国际航线(加港澳台地区)':'总旅客量'},inplace=True)

predict_amount = predict_domestic + predict_foreign


true_amount = amount.loc[:'Jun-20']
plt.figure(figsize=(16,8))
plt.title('总旅客量同比变化(假设未来12个月无疫情反弹)')
# plt.xticks(rotation=300)
# plt.plot(train, label='Train')
plt.plot(true_amount, label='pre_COVID')
# plt.plot(y_hat_avg['Holt_Winter'], label='Holt_Winter')
plt.plot(predict_amount, label='predict_amount')
plt.legend(loc='best')
plt.show()






# =================================== 暑期反弹 ================================================
change_rate = np.arange(19)
change_rate[0] = (predict_2.values[0]+predict_2.values[1]+predict_2.values[2]+predict_2.values[3]+predict_2.values[4])/5
change_rate[1] = (predict_2.values[5]+predict_2.values[6]+predict_2.values[7]+predict_2.values[8])/4
change_rate[2] = (predict_2.values[9]+predict_2.values[10]+predict_2.values[11]+predict_2.values[12])/4
change_rate[3] =  (predict_2.values[13]+predict_2.values[14]+predict_2.values[15]+predict_2.values[16]+predict_2.values[17])/5
change_rate[4] = (predict_2.values[18]+predict_2.values[19]+predict_2.values[20]+predict_2.values[21])/4
change_rate[5] = (predict_2.values[22]+predict_2.values[23]+predict_2.values[24]+predict_2.values[25])/4
change_rate[6] = (predict_2.values[26]+predict_2.values[27]+predict_2.values[28]+predict_2.values[29])/4
change_rate[7] = (predict_2.values[30]+predict_2.values[31]+predict_2.values[32]+predict_2.values[33])/4
change_rate[8] = (predict_2.values[34]+predict_2.values[35]+predict_2.values[36]+predict_2.values[37])/4
change_rate[9] = (predict_2.values[38]+predict_2.values[39]+predict_2.values[40]+predict_2.values[41])/4
change_rate[10] = (predict_2.values[42]+predict_2.values[43]+predict_2.values[44]+predict_2.values[45])/4
change_rate[11] = (predict_2.values[46]+predict_2.values[47]+predict_2.values[48]+predict_2.values[49])/4
change_rate[12] = (predict_2.values[50]+predict_2.values[51]+predict_2.values[52]+predict_2.values[53])/4
change_rate[13] = (predict_2.values[54]+predict_2.values[55]+predict_2.values[56]+predict_2.values[57])/4
change_rate[14] = (predict_2.values[58]+predict_2.values[59]+predict_2.values[60]+predict_2.values[61])/4
change_rate[15] = (predict_2.values[62]+predict_2.values[63]+predict_2.values[64]+predict_2.values[65])/4
change_rate[16] = (predict_2.values[66]+predict_2.values[67]+predict_2.values[68]+predict_2.values[69])/4
change_rate[17] = (predict_2.values[70]+predict_2.values[71]+predict_2.values[72]+predict_2.values[73])/4
change_rate[18] = (predict_2.values[74]+predict_2.values[75]+predict_2.values[76]+predict_2.values[77])/4

post_COVID = domestic.loc['Dec-19':]

predict_val = (post_COVID_domestic.values[1,:].T * (change_rate.T / 100)) + post_COVID_domestic.values[1,:].T

predict_amount = post_COVID.copy()
predict_amount.loc['Jun-20',:] = domestic.loc['Jun-20']
predict_amount.loc['Jul-20',:] = predict_val[6]
predict_amount.loc['Aug-20',:] = predict_val[7]
predict_amount.loc['Sep-20',:] = predict_val[8]
predict_amount.loc['Oct-20',:] = predict_val[9]
predict_amount.loc['Nov-20',:] = predict_val[10]
predict_amount.loc['Dec-20',:] = predict_val[11]
predict_amount.loc['Jan-21',:] = predict_val[12]
predict_amount.loc['Feb-21',:] = predict_val[13]
predict_amount.loc['Mar-21',:] = predict_val[14]
predict_amount.loc['Apr-21',:] = predict_val[15]
predict_amount.loc['May-21',:] = predict_val[16]
predict_amount.loc['Jun-21',:] = predict_val[17]
predict_amount.loc['Jul-21',:] = predict_val[18]
predict_domestic = predict_amount.loc['Jun-20':]
predict_domestic = predict_domestic.copy()
predict_domestic.rename(columns={'国内航线(大陆)':'总旅客量'},inplace=True)


change_rate = np.arange(19)
change_rate[0] = (predict_5.values[0]+predict_5.values[1]+predict_5.values[2]+predict_5.values[3]+predict_5.values[4])/5
change_rate[1] = (predict_5.values[5]+predict_5.values[6]+predict_5.values[7]+predict_5.values[8])/4
change_rate[2] = (predict_5.values[9]+predict_5.values[10]+predict_5.values[11]+predict_5.values[12])/4
change_rate[3] =  (predict_5.values[13]+predict_5.values[14]+predict_5.values[15]+predict_5.values[16]+predict_5.values[17])/5
change_rate[4] = (predict_5.values[18]+predict_5.values[19]+predict_5.values[20]+predict_5.values[21])/4
change_rate[5] = (predict_5.values[22]+predict_5.values[23]+predict_5.values[24]+predict_5.values[25])/4
change_rate[6] = (predict_5.values[26]+predict_5.values[27]+predict_5.values[28]+predict_5.values[29])/4
change_rate[7] = (predict_5.values[30]+predict_5.values[31]+predict_5.values[32]+predict_5.values[33])/4
change_rate[8] = (predict_5.values[34]+predict_5.values[35]+predict_5.values[36]+predict_5.values[37])/4
change_rate[9] = (predict_5.values[38]+predict_5.values[39]+predict_5.values[40]+predict_5.values[41])/4
change_rate[10] = (predict_5.values[42]+predict_5.values[43]+predict_5.values[44]+predict_5.values[45])/4
change_rate[11] = (predict_5.values[46]+predict_5.values[47]+predict_5.values[48]+predict_5.values[49])/4
change_rate[12] = (predict_5.values[50]+predict_5.values[51]+predict_5.values[52]+predict_5.values[53])/4
change_rate[13] = (predict_5.values[54]+predict_5.values[55]+predict_5.values[56]+predict_5.values[57])/4
change_rate[14] = (predict_5.values[58]+predict_5.values[59]+predict_5.values[60]+predict_5.values[61])/4
change_rate[15] = (predict_5.values[62]+predict_5.values[63]+predict_5.values[64]+predict_5.values[65])/4
change_rate[16] = (predict_5.values[66]+predict_5.values[67]+predict_5.values[68]+predict_5.values[69])/4
change_rate[17] = (predict_5.values[70]+predict_5.values[71]+predict_5.values[72]+predict_5.values[73])/4
change_rate[18] = (predict_5.values[74]+predict_5.values[75]+predict_5.values[76]+predict_5.values[77])/4

post_COVID = foreign.loc['Dec-19':]

predict_val = (post_COVID_foreign.values[1,:].T * (change_rate.T / 100)) + post_COVID_foreign.values[1,:].T

predict_amount = post_COVID.copy()
predict_amount.loc['Jun-20',:] = foreign.loc['Jun-20']
predict_amount.loc['Jul-20',:] = predict_val[6]
predict_amount.loc['Aug-20',:] = predict_val[7]
predict_amount.loc['Sep-20',:] = predict_val[8]
predict_amount.loc['Oct-20',:] = predict_val[9]
predict_amount.loc['Nov-20',:] = predict_val[10]
predict_amount.loc['Dec-20',:] = predict_val[11]
predict_amount.loc['Jan-21',:] = predict_val[12]
predict_amount.loc['Feb-21',:] = predict_val[13]
predict_amount.loc['Mar-21',:] = predict_val[14]
predict_amount.loc['Apr-21',:] = predict_val[15]
predict_amount.loc['May-21',:] = predict_val[16]
predict_amount.loc['Jun-21',:] = predict_val[17]
predict_amount.loc['Jul-21',:] = predict_val[18]
predict_foreign = predict_amount.loc['Jun-20':]
predict_foreign = predict_foreign.copy()
predict_foreign.rename(columns={'国际航线(加港澳台地区)':'总旅客量'},inplace=True)

predict_amount = predict_domestic + predict_foreign


true_amount = amount.loc[:'Jun-20']
plt.figure(figsize=(16,8))
plt.title('总旅客量同比变化(假设今年暑期疫情反弹)')
# plt.xticks(rotation=300)
# plt.plot(train, label='Train')
plt.plot(true_amount, label='pre_COVID')
# plt.plot(y_hat_avg['Holt_Winter'], label='Holt_Winter')
plt.plot(predict_amount, label='predict_amount')
plt.legend(loc='best')
plt.show()







# =================================== 冬季反弹 ================================================
change_rate = np.arange(19)
change_rate[0] = (predict_3.values[0]+predict_3.values[1]+predict_3.values[2]+predict_3.values[3]+predict_3.values[4])/5
change_rate[1] = (predict_3.values[5]+predict_3.values[6]+predict_3.values[7]+predict_3.values[8])/4
change_rate[2] = (predict_3.values[9]+predict_3.values[10]+predict_3.values[11]+predict_3.values[12])/4
change_rate[3] =  (predict_3.values[13]+predict_3.values[14]+predict_3.values[15]+predict_3.values[16]+predict_3.values[17])/5
change_rate[4] = (predict_3.values[18]+predict_3.values[19]+predict_3.values[20]+predict_3.values[21])/4
change_rate[5] = (predict_3.values[22]+predict_3.values[23]+predict_3.values[24]+predict_3.values[25])/4
change_rate[6] = (predict_3.values[26]+predict_3.values[27]+predict_3.values[28]+predict_3.values[29])/4
change_rate[7] = (predict_3.values[30]+predict_3.values[31]+predict_3.values[32]+predict_3.values[33])/4
change_rate[8] = (predict_3.values[34]+predict_3.values[35]+predict_3.values[36]+predict_3.values[37])/4
change_rate[9] = (predict_3.values[38]+predict_3.values[39]+predict_3.values[40]+predict_3.values[41])/4
change_rate[10] = (predict_3.values[42]+predict_3.values[43]+predict_3.values[44]+predict_3.values[45])/4
change_rate[11] = (predict_3.values[46]+predict_3.values[47]+predict_3.values[48]+predict_3.values[49])/4
change_rate[12] = (predict_3.values[50]+predict_3.values[51]+predict_3.values[52]+predict_3.values[53])/4
change_rate[13] = (predict_3.values[54]+predict_3.values[55]+predict_3.values[56]+predict_3.values[57])/4
change_rate[14] = (predict_3.values[58]+predict_3.values[59]+predict_3.values[60]+predict_3.values[61])/4
change_rate[15] = (predict_3.values[62]+predict_3.values[63]+predict_3.values[64]+predict_3.values[65])/4
change_rate[16] = (predict_3.values[66]+predict_3.values[67]+predict_3.values[68]+predict_3.values[69])/4
change_rate[17] = (predict_3.values[70]+predict_3.values[71]+predict_3.values[72]+predict_3.values[73])/4
change_rate[18] = (predict_3.values[74]+predict_3.values[75]+predict_3.values[76]+predict_3.values[77])/4

post_COVID = domestic.loc['Dec-19':]

predict_val = (post_COVID_domestic.values[1,:].T * (change_rate.T / 100)) + post_COVID_domestic.values[1,:].T

predict_amount = post_COVID.copy()
predict_amount.loc['Jun-20',:] = domestic.loc['Jun-20']
predict_amount.loc['Jul-20',:] = predict_val[6]
predict_amount.loc['Aug-20',:] = predict_val[7]
predict_amount.loc['Sep-20',:] = predict_val[8]
predict_amount.loc['Oct-20',:] = predict_val[9]
predict_amount.loc['Nov-20',:] = predict_val[10]
predict_amount.loc['Dec-20',:] = predict_val[11]
predict_amount.loc['Jan-21',:] = predict_val[12]
predict_amount.loc['Feb-21',:] = predict_val[13]
predict_amount.loc['Mar-21',:] = predict_val[14]
predict_amount.loc['Apr-21',:] = predict_val[15]
predict_amount.loc['May-21',:] = predict_val[16]
predict_amount.loc['Jun-21',:] = predict_val[17]
predict_amount.loc['Jul-21',:] = predict_val[18]
predict_domestic = predict_amount.loc['Jun-20':]
predict_domestic = predict_domestic.copy()
predict_domestic.rename(columns={'国内航线(大陆)':'总旅客量'},inplace=True)


change_rate = np.arange(19)
change_rate[0] = (predict_6.values[0]+predict_6.values[1]+predict_6.values[2]+predict_6.values[3]+predict_6.values[4])/5
change_rate[1] = (predict_6.values[5]+predict_6.values[6]+predict_6.values[7]+predict_6.values[8])/4
change_rate[2] = (predict_6.values[9]+predict_6.values[10]+predict_6.values[11]+predict_6.values[12])/4
change_rate[3] =  (predict_6.values[13]+predict_6.values[14]+predict_6.values[15]+predict_6.values[16]+predict_6.values[17])/5
change_rate[4] = (predict_6.values[18]+predict_6.values[19]+predict_6.values[20]+predict_6.values[21])/4
change_rate[5] = (predict_6.values[22]+predict_6.values[23]+predict_6.values[24]+predict_6.values[25])/4
change_rate[6] = (predict_6.values[26]+predict_6.values[27]+predict_6.values[28]+predict_6.values[29])/4
change_rate[7] = (predict_6.values[30]+predict_6.values[31]+predict_6.values[32]+predict_6.values[33])/4
change_rate[8] = (predict_6.values[34]+predict_6.values[35]+predict_6.values[36]+predict_6.values[37])/4
change_rate[9] = (predict_6.values[38]+predict_6.values[39]+predict_6.values[40]+predict_6.values[41])/4
change_rate[10] = (predict_6.values[42]+predict_6.values[43]+predict_6.values[44]+predict_6.values[45])/4
change_rate[11] = (predict_6.values[46]+predict_6.values[47]+predict_6.values[48]+predict_6.values[49])/4
change_rate[12] = (predict_6.values[50]+predict_6.values[51]+predict_6.values[52]+predict_6.values[53])/4
change_rate[13] = (predict_6.values[54]+predict_6.values[55]+predict_6.values[56]+predict_6.values[57])/4
change_rate[14] = (predict_6.values[58]+predict_6.values[59]+predict_6.values[60]+predict_6.values[61])/4
change_rate[15] = (predict_6.values[62]+predict_6.values[63]+predict_6.values[64]+predict_6.values[65])/4
change_rate[16] = (predict_6.values[66]+predict_6.values[67]+predict_6.values[68]+predict_6.values[69])/4
change_rate[17] = (predict_6.values[70]+predict_6.values[71]+predict_6.values[72]+predict_6.values[73])/4
change_rate[18] = (predict_6.values[74]+predict_6.values[75]+predict_6.values[76]+predict_6.values[77])/4

post_COVID = foreign.loc['Dec-19':]

predict_val = (post_COVID_foreign.values[1,:].T * (change_rate.T / 100)) + post_COVID_foreign.values[1,:].T

predict_amount = post_COVID.copy()
predict_amount.loc['Jun-20',:] = foreign.loc['Jun-20']
predict_amount.loc['Jul-20',:] = predict_val[6]
predict_amount.loc['Aug-20',:] = predict_val[7]
predict_amount.loc['Sep-20',:] = predict_val[8]
predict_amount.loc['Oct-20',:] = predict_val[9]
predict_amount.loc['Nov-20',:] = predict_val[10]
predict_amount.loc['Dec-20',:] = predict_val[11]
predict_amount.loc['Jan-21',:] = predict_val[12]
predict_amount.loc['Feb-21',:] = predict_val[13]
predict_amount.loc['Mar-21',:] = predict_val[14]
predict_amount.loc['Apr-21',:] = predict_val[15]
predict_amount.loc['May-21',:] = predict_val[16]
predict_amount.loc['Jun-21',:] = predict_val[17]
predict_amount.loc['Jul-21',:] = predict_val[18]
predict_foreign = predict_amount.loc['Jun-20':]
predict_foreign = predict_foreign.copy()
predict_foreign.rename(columns={'国际航线(加港澳台地区)':'总旅客量'},inplace=True)

predict_amount = predict_domestic + predict_foreign


true_amount = amount.loc[:'Jun-20']
plt.figure(figsize=(16,8))
plt.title('总旅客量同比变化(假设今年冬季疫情反弹)')
# plt.xticks(rotation=300)
# plt.plot(train, label='Train')
plt.plot(true_amount, label='pre_COVID')
# plt.plot(y_hat_avg['Holt_Winter'], label='Holt_Winter')
plt.plot(predict_amount, label='predict_amount')
plt.legend(loc='best')
plt.show()

















