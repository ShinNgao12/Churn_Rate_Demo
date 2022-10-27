import pandas as pd
import numpy as np
import pickle

from google.cloud import bigquery

from datetime import datetime as dt
from datetime import date, timedelta
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report, roc_auc_score, plot_confusion_matrix
from sklearn.metrics import auc, plot_roc_curve, precision_recall_curve

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.float_format', lambda x: '%.2f' % x)


day_pre_interval = '2022-03-25' # edit 

# read data
df_gcp_raw = pd.read_csv('data_gcp_0712.csv') # edit 


# convert string to datetime
df_gcp = df_gcp_raw

df_gcp['first_trans_date'] = pd.to_datetime(df_gcp['first_trans_date'])
df_gcp['last_trans_date'] = pd.to_datetime(df_gcp['last_trans_date'])

# calculate purchase value and avg point
df_gcp['purchase_val'] = df_gcp['purchase_val'].astype(float).apply(lambda x: int(x))
df_gcp['avg_available_point'] = df_gcp['avg_available_point'].astype(float).apply(lambda x: int(x))

# drop missing value --> 
# df_gcp.dropna(
#     subset = ['job','gender','province_1'],
#     how = 'any',
#     inplace =  True
# )

# convert datetime to distance between 2 transactions 
df_gcp['days_from_last_trans'] = (pd.to_datetime(day_pre_interval) - df_gcp["last_trans_date"]).dt.days.astype(int)
df_gcp['days_from_first_trans'] = (pd.to_datetime(day_pre_interval) - df_gcp['first_trans_date']).dt.days.astype(int)

# convert gender to cate
df_gcp['gender'].replace([1,2,3], ['Female', 'Male', 'Unknow'], inplace=True)

interval_churn_cutoff = 201.0

# label
df_gcp['churn'] = np.where(df_gcp['days_from_last_trans'] > interval_churn_cutoff, 1, 0)


###################### PREPROCESSING DATA #####################

font = {'size' : 18}
plt.rc('font', **font)

sns.set_theme(rc={'figure.figsize': (14, 8)})

df_X = df_gcp.drop(
    columns = [
        'sf_loyalty_number_id','first_trans_date', 'last_trans_date','days_from_last_trans', 'churn'
    ]
)

y = df_gcp['churn']

# One hot categorical featutes
category_col = df_X.select_dtypes('O').columns
df_X_dummies = pd.get_dummies(df_X[category_col], dummy_na = True).reset_index().drop(columns= 'index')

# Scaling numerical features
numeric_col  = df_X.select_dtypes(['float64','int64']).columns
scaler = StandardScaler()
scaler.fit(df_X[numeric_col])
df_X_scale = pd.DataFrame(scaler.transform(df_X[numeric_col]), columns = numeric_col).reset_index().drop(columns= 'index')

# Concat data
X = pd.concat((df_X_scale, df_X_dummies), axis = 1) 

# oversampling
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X,y)

###################### BUILD MODEL #####################
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=test_size, 
                                                    random_state=42, # chia ngau nhien theo cung mot cach (giua cac lan chay)
                                                    stratify=y)  # what is stratify, random_state ???

logreg = LogisticRegression()
logreg.fit(X=X_train, y=y_train)

pickle.dump(logreg, open('churn_rate_logreg.pkl', 'wb'))