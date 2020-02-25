# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 22:27:28 2020

@author: orastak
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 12:08:47 2020

@author: orastak
"""

#%%getting daily production plan data
import pandas as pd
df11 = pd.io.excel.read_excel(r"C:\Users\orastak\Desktop\excels for PTF forecast\dpp2010-2013.12.xlsx", sheetname=0)
df12 = pd.io.excel.read_excel(r"C:\Users\orastak\Desktop\excels for PTF forecast\dpp2014-2017.12.xlsx", sheetname=0)
df13 = pd.io.excel.read_excel(r"C:\Users\orastak\Desktop\excels for PTF forecast\dpp2018.12.xlsx", sheetname=0)
df_dpp=pd.concat([df11,df12,df13])
import datetime
df_dpp['date']=[i.replace('T',' ') for i in df_dpp['date']]
df_dpp['date']=[i.split('.',1)[0] for i in df_dpp['date']]
df_dpp['date']=[datetime.datetime.strptime(i, "%Y-%m-%d %H:%M:%S") for i in df_dpp['date']]
#%% getting Market clearing price data
df_mcp = pd.io.excel.read_excel(r"C:\Users\orastak\Desktop\excels for PTF forecast\mcp_smp.xlsx", sheetname=0)
# fixing date format
df_mcp['date']=[i.replace('T',' ') for i in df_mcp['date']]
df_mcp['date']=[i.split('.',1)[0] for i in df_mcp['date']]
df_mcp['date']=[datetime.datetime.strptime(i, "%Y-%m-%d %H:%M:%S") for i in df_mcp['date']]
#%% getting load estimation plan via API
import numpy as np
import pandas as pd
import http.client
import json
import time
conn = http.client.HTTPSConnection("api.epias.com.tr")
headers = {
    'x-ibm-client-id': "...",
    'accept': "application/json"
    }
conn.request("GET", "epias/exchange/transparency/consumption/load-estimation-plan?startDate=2010-01-01T00:00&endDate=2018-12-31T00:00", headers=headers)
res = conn.getresponse()
data5 = res.read()
data6=json.loads(data5)
df_load=pd.DataFrame.from_dict(data6["body"]["loadEstimationPlanList"])
# fixing date format
df_load['date']=[i.replace('T',' ') for i in df_load['date']]
df_load['date']=[i.split('.',1)[0] for i in df_load['date']]
df_load['date']=[datetime.datetime.strptime(i, "%Y-%m-%d %H:%M:%S") for i in df_load['date']]
#%% merge dataframes with date key
df_mcp_dpp=pd.merge(df_mcp, df_dpp, on='date', how='inner')
df_mcp_dpp_load=pd.merge(df_mcp_dpp, df_load, on='date', how='inner')
#%% extract month, year, hour as a column for filtering later
df_mcp_dpp_load['month'] = pd.DatetimeIndex(df_mcp_dpp_load['date']).month
df_mcp_dpp_load['year'] = pd.DatetimeIndex(df_mcp_dpp_load['date']).year
df_mcp_dpp_load['hour'] = pd.DatetimeIndex(df_mcp_dpp_load['date']).hour
#%% drop column mcpState
df_mcp_dpp_load=df_mcp_dpp_load.drop(columns=['mcpState'])
#%% insert to SQL
import pandas as pd
from sqlalchemy import create_engine
engine = create_engine("mssql+pyodbc://user:password@servername/tablename?driver=SQL+Server+Native+Client+11.0")
df_mcp_dpp_load.to_sql('MCP_DPP_Load', con = engine, if_exists = 'replace')
#%%  Histogram for checking distribution
import matplotlib.pyplot as plt
df_train.iloc[:,1].plot.hist(bins=18, alpha=0.5)
plt.show()
# histogram after log tranformation applied
np.log(df_train.iloc[:,1]).plot.hist(bins=18, alpha=0.5)
plt.show()
#%% Z-score outlier detector
import numpy as np
import pandas as pd
def Zscore_outlier_detector(data_1): 
    outliers=[]
    threshold=3
    mean_1 = np.mean(data_1)
    std_1 =np.std(data_1)
    
    for y in data_1:
        z_score= (y - mean_1)/std_1 
        if np.abs(z_score) > threshold:
            outliers.append(y)
    return outliers
#%% IQR outliars
def IQR_outlier_detector(data):
    data.describe()
    Q1=np.percentile(data, 25)
    Q3=np.percentile(data, 75)
    IQR=Q3-Q1
    lower_range=Q1-(1.5*IQR)    
    upper_range=Q3+(1.5*IQR) 
    outliers=df_train.iloc[(data<lower_range).values|(data>upper_range).values,1]
    outliers=outliers.tolist()
    return outliers
#%%
outliars=[]
for column in df_mcp_dpp_load: 
    outliars.append(Zscore_outlier_detector(df_mcp_dpp_load.iloc[:,i]))
outliars
#%% 8 to 22 hours are high demand times lets filter 
df_mcp_dpp_load2015=df_mcp_dpp_load[(df_mcp_dpp_load.year>=2015)&(df_mcp_dpp_load.hour>=8)&(df_mcp_dpp_load.hour<=22)].reset_index(drop=True)
from sklearn.preprocessing import StandardScaler
#%% 
features = df_mcp_dpp_load2015.drop(columns=["smpDirection", "hour","month","year","smp","date","mcp","mcpState","biomass"])
# Separating out the features
x = features
# Separating out the target
y = df_mcp_dpp_load2015['mcp']
#%%
"""
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

finalDf = pd.concat([principalDf, y], axis = 1)
#%%
import matplotlib.pyplot as plt
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = ['r', 'g', 'b']
for mcp, color in zip(targets,colors):
    indicesToKeep = finalDf['mcp'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()

from sklearn.feature_selection import SelectKBest, chi2
"""
#%%

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.33, random_state=101)

#%% XGBOOST
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
scaler=MinMaxScaler(feature_range=(0,1))
#prepare the model XGB
x_train_scaled=scaler.fit_transform(x_train)
xg_reg=xgb.XGBRegressor(objective="reg:squarederror",colsample_bytree
=0.3,learning_rate=0.03, max_depth=5, alpha=10, n_estimators=100)

xg_reg.fit(x_train_scaled, y_train)
train_pred=xg_reg.predict(x_train_scaled)
rmse=np.sqrt(mean_squared_error(y_train, train_pred))
print("RMSE_train: %f" % (rmse))

x_test_scaled=scaler.transform(x_test)
test_pred=xg_reg.predict(x_test_scaled)
rmse=np.sqrt(mean_squared_error(y_test, test_pred))
print("RMSE_test: %f" % (rmse))
#%%
def rmsle(train_pred, y_test):
    error=np.square(np.log10(train_pred+1)-np.log10(y_test+1)).mean()**0.5
    Acc=1-error
    return Acc

print("Accuracy attained on Training Set=", rmsle(train_pred, y_train))
print("Accuracy attained on Test Set=", rmsle(test_pred, y_test))
#%%
fig=plt.figure(figsize=(15,6))
plt.xticks(rotation="horizontal")
plt.bar([i for i in range(len(xg_reg.feature_importances_))], xg_reg.feature_importances_,
         tick_label=x_test.columns, color="chocolate")
plt.show()
#%%
matrix=xgb.DMatrix(data=x, label=y)
params={"objective":"reg:squarederror","colcsmaple_bytree":0.3, "learning_rate":0.1,
        "max_depth":5, "alpha":10}
cv_results=xgb.cv(dtrain=matrix, params=params, nfold=5, num_boost_round=50,
                  early_stopping_rounds=10, metrics="rmse", as_pandas=True, seed=123)
print((cv_results["test-rmse-mean"]).tail(1))