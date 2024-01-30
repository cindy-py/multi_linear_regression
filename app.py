#import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


#load the dataset
df=pd.read_excel("Dropoutdataset.xlsx")


#provide title
st.title("Multiple Linear Regression App")

#display more EDA info of the dataset
#rows are represented by [0] and columns are represented by [1]
st.write("Dataset info:", df.info())
st.write("Number of rows in the dataset", df.shape[0])
st.write("Number of columns in the dataset", df.shape[1])
st.write("Column names in the dataset", df.columns.tolist())
st.write("Datatypes in the dataset", df.dtypes)
st.write("Null values in the dataset", df.isnull())
st.write("Sum of null values in the dataset", df.isnull().sum())

#button to trigger
#plot barchart
if st.button("Generate Bar Chart"):
    selected_columns = st.multiselect("Select the Columns to Visualize the Bar", df.columns)
    if selected_columns:
        st.bar_chart(df[selected_columns])
    else:
        st.warning("Select atleast two columns.")
        
#button for linechart
if st.button("Generate Line Chart"):
    selected_columns = st.multiselect("Select the Columns to Visualize the Linechart", df.columns)
    if selected_columns:
        st.line_chart(df[selected_columns])
    else:
        st.warning("Select atleast two columns.")
        
#button for Histogram
if st.button("Generate Histogram"):
    selected_columns = st.multiselect("Select the Columns to Visualize the Histogram", df.columns)
    if selected_columns:
        st.histogram(df[selected_columns])
    else:
        st.warning("Select atleast two columns.")
        
        
#Feature Engineering
   #Encoding the target column using label encoder
university = LabelEncoder()
df['Target'] = university.fit_transform(df['Target'])
#use OneHotEncoder to encode the categorical features
ct = ColumnTransformer(transformers=[('encode', OneHotEncoder(), ['Target'])] , remainder="passthrough")
x = df.iloc[:,: -1]
y = df.iloc[:, -1]
y_encoded = ct.fit_transform(df[['Target']])
#split the dataset into testing and training
x_train , x_test , y_train , Y_test = train_test_split(x,y_encoded,test_size=0.2,random_state=0)
#fit the regression model
regressor = LinearRegression()
regressor.fit(x_train,y_train)
#use the input for independent variables
st.sidebar.title("Enter values to be predicted")
#create the input for each feature
user_input = {}
for feature in df.columns[:-1]:
    user_input[feature] = st.sidebar.text_input(f"Enter {feature}", 0.0 )
    
#button to trigger the prediction
if st.sidebar.button("Predict"):
    #create dataframe for user input
    user_input_df = pd.DataFrame([user_input], dtype=float)
    #predict the trained model
    y_pred = regressor.predict(user_input_df)
    #inverse transform to get the original target values
    predicted_class = university.inverse_transform(np.array(y_pred, axis=1))
    #display the predicted class/target
    st.write('### predicted class')
    st.write(predicted_class[0])
