import streamlit as st
import pandas as pd
import numpy as np
from datetime import date ,datetime
import pickle

from sklearn.preprocessing import StandardScaler

st.write("""
# Churn Rate Application with Binary Classification 
""")

st.sidebar.header('User Input Features')
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

day_pre_interval = st.sidebar.date_input('Day previous interval', datetime.now())
day_pre_interval = '2022-04-07'

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        purchase_val = st.sidebar.number_input('Purchase value', format='%d',
                                               min_value = 0, step = 1)
        first_trans_date = st.sidebar.date_input('First transaction date', datetime.now())
        last_trans_date = st.sidebar.date_input('Last transaction date', datetime.now())
        count_trans = st.sidebar.number_input('Count transaction', format = '%d',
                                              min_value = 0, step = 1)
        job =  st.sidebar.selectbox('Job',
                            ('Hưu trí','Kinh doanh Online/ Kinh doanh tự do/ Tự làm chủ',
                            'Lao động phổ thông ngoài trời', 'Nhân viên VP', 
                            'Ngành nghề chuyên môn (Bác sĩ, Dược sĩ, Kỹ sư, ...)',
                            'Giảng viên/ Giáo Viên', 'Nội trợ',
                            'Quản lý cấp trung (Trưởng phòng, Phó phòng/Trưởng nhóm..)',
                            'Quản lý cấp cao (Giám đốc, Phó GĐ)', 'Học sinh - sinh viên',
                            'Công nhân trong nhà máy', 'Nhân viên Sales','Khác', np.nan))
        gender = st.sidebar.selectbox('Gender: Female(1), Male(2), Unknow(3)', (1,2,3))
        province = st.sidebar.selectbox('Province',('Thành Phố Hà Nội', 'Thành Phố Hồ Chí Minh', 'Tỉnh Thanh Hóa',
                                'Tỉnh Đồng Tháp', 'Tỉnh Vĩnh Long', 'Tỉnh Bình Định',
                                'Tỉnh Bình Dương', 'Tỉnh Tiền Giang', 'Tỉnh Bắc Ninh',
                                'Tỉnh Nghệ An', 'Tỉnh Hà Nam', 'Tỉnh Yên Bái', 'Tỉnh Ninh Thuận',
                                'Tỉnh Quảng Nam', 'Tỉnh An Giang', 'Thành Phố Cần Thơ',
                                'Tỉnh Vĩnh Phúc', 'Tỉnh Long An', 'Tỉnh Quảng Ninh',
                                'Tỉnh Hưng Yên', 'Tỉnh Thừa Thiên Huế', 'Tỉnh Gia Lai',
                                'Tỉnh Hà Tĩnh', 'Tỉnh Kon Tum', 'Tỉnh Sóc Trăng',
                                'Tỉnh Bình Thuận', 'Tỉnh Đồng Nai', 'Tỉnh Thái Nguyên',
                                'Tỉnh Kiên Giang', 'Tỉnh Đắk Lắk', 'Tỉnh Bà Rịa - Vũng Tàu',
                                'Tỉnh Bến Tre', 'Tỉnh Thái Bình', 'Tỉnh Bạc Liêu',
                                'Tỉnh Quảng Bình', 'Tỉnh Hà Giang', 'Tỉnh Lào Cai',
                                'Tỉnh Bắc Giang', 'Tỉnh Phú Thọ', 'Tỉnh Hải Dương',
                                'Thành Phố Đà Nẵng', 'Tỉnh Quảng Ngãi', 'Tỉnh Phú Yên',
                                'Tỉnh Khánh Hòa', 'Tỉnh Lâm Đồng', 'Tỉnh Tây Ninh',
                                'Tỉnh Trà Vinh', 'Tỉnh Hậu Giang', 'Tỉnh Cà Mau', 'Tỉnh Cao Bằng',
                                'Tỉnh Tuyên Quang', 'Thành Phố Hải Phòng', 'Tỉnh Nam Định',
                                'Tỉnh Sơn La', 'Tỉnh Ninh Bình', 'Tỉnh Bình Phước',
                                'Tỉnh Quảng Trị', 'Tỉnh Điện Biên', 'Tỉnh Đắk Nông',
                                'Tỉnh Bắc Kạn', 'Tỉnh Lạng Sơn', 'Tỉnh Lai Châu', 'Tỉnh Hòa Bình', np.nan))
        avg_available_point = st.sidebar.number_input('Average available points', format='%d',
                                                      min_value = 0, step = 1)
        
        data = {'purchase_val': purchase_val,
                'first_trans_date': first_trans_date,
                'last_trans_date': last_trans_date,
                'count_trans': count_trans,
                'job': job,
                'province_1': province,
                'avg_available_point': avg_available_point
                }
        
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

# read raw data to transform
df = pd.read_csv('data_gcp_0712.csv')

# drop missing value 
df.dropna(
    subset = ['job','gender','province_1'],
    how = 'any',
    inplace =  True
)

# concat data
df_gcp = pd.concat((df, input_df), axis = 0)

df_gcp['first_trans_date'] = pd.to_datetime(df_gcp['first_trans_date'])
df_gcp['last_trans_date'] = pd.to_datetime(df_gcp['last_trans_date'])

# calculate purchase value and avg point
df_gcp['purchase_val'] = df_gcp['purchase_val'].astype(float).apply(lambda x: int(x))
df_gcp['avg_available_point'] = df_gcp['avg_available_point'].astype(float).apply(lambda x: int(x))

# convert gender to cate
df_gcp['gender'].replace([1,2,3], ['Female', 'Male', 'Unknow'], inplace=True)

day_pre_interval = '2022-04-07'

# convert datetime to distance between 2 transactions 
df_gcp['days_from_last_trans'] = (pd.to_datetime(day_pre_interval) - df_gcp["last_trans_date"]).dt.days.astype(int)
df_gcp['days_from_first_trans'] = (pd.to_datetime(day_pre_interval) - df_gcp['first_trans_date']).dt.days.astype(int)

df_X = df_gcp.drop(
    columns = [
        'sf_loyalty_number_id','first_trans_date', 'last_trans_date','days_from_last_trans'
    ]
)

# categorical features
category_col = df_X.select_dtypes('O').columns
df_X_dummies = pd.get_dummies(df_X[category_col], dummy_na = True).reset_index().drop(columns= 'index')

# numerical features
numeric_col  = df_X.select_dtypes(['float64','int64']).columns
scaler = StandardScaler()
scaler.fit(df_X[numeric_col])
df_X_scale = pd.DataFrame(scaler.transform(df_X[numeric_col]), columns = numeric_col).reset_index().drop(columns= 'index')

X = pd.concat((df_X_scale, df_X_dummies), axis = 1) 
input_scale = X.iloc[df.shape[0]:,:]

# show input data
st.sidebar.subheader('Day previous interval')
st.write(day_pre_interval)

st.subheader('User Input features')

if uploaded_file is not None:
    st.write(input_df)
    
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(input_df)


logreg = pickle.load(open('churn_rate_logreg.pkl', 'rb'))

prediction_proba = logreg.predict_proba(input_scale)
result  = pd.DataFrame(logreg.predict(input_scale), columns = ['Label'])
result.replace((1,0), ('Churn', 'Not churn'), inplace = True)
result['probability'] = np.max(prediction_proba, axis = 1)

st.subheader('Prediction')
st.write(result)

