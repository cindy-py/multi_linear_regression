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

#load the image
st.image("Africdsa.jpeg")

#add title to app
st.title("Multiple Linear Regression App")

#add header
st.header("Dataset Concept." , divider="rainbow")

#add paragraph
st.write("""The Dropout dataset is a comprehensive collection of information related to students' academic performance and various socio-economic factors, 
            aimed at understanding the factors influencing students decisions to either graduate, dropout, or remain enrolled in educational institutions.
            This dataset includes features such as socio-economic background, parental education, academic scores, attendance,and extracurricular activities.
            In the context of multi-linear regression, researchers and 
            data scientists utilize this dataset to build predictive models that can assess the likelihood of a student either graduating, 
            dropping out, or remaining enrolled based on a combination of these factors. By employing multi-linear regression techniques, 
            the dataset allows for the examination of the relationships and interactions among multiple independent variables simultaneously. 
            The model seeks to identify which specific factors play a significant role in predicting the educational outcomes of students, 
            providing valuable insights for educators, policymakers, and institutions to implement targeted interventions and support systems for at-risk students. 
            Through the analysis of the Dropout dataset, it becomes possible to develop more informed strategies to improve overall student success and reduce dropout rates."""
)

#----------------- display our EDA ---------------
st.header("Exploratory Data Analysis(EDA)." , divider="rainbow")

if st.checkbox("Dataset info"):
    st.write("Dataset info", df.info())
    
if st.checkbox("Number of rows"):
    st.write("Number of rows:", df.shape[0])
    
if st.checkbox("Column names"):
    st.write("Column names:", df.columns.tolist())
    
if st.checkbox("Data types"):
    st.write("Data types:", df.dtypes)
    
if st.checkbox("Missing values"):
    st.write("Missing values:", df.isnull().sum())
    
if st.checkbox("Statistical Summary"):
    st.write("Statistical Summary:", df.describe())

#-------------visaulization-----------------
st.header("Visualization of the Dataset." , divider="rainbow")

#barchart
if st.checkbox("Inflation against GDP Barchart"):
    st.write("Barchart of inflation against GDP")
    st.bar_chart(x="Inflation rate" , y="GDP" , data=df , color=["#FF0000"])
    
#linechart
if st.checkbox("Inflation against GDP Linechart"):
    st.write("Linechart of inflation against GDP")
    st.line_chart(x="Inflation rate" , y="GDP" , data=df , color=["#FF0000"])
    
#scatterplot
if st.checkbox("Inflation against GDP Scatterplot"):
    st.write("Scatterplot of inflation against GDP")
    st.scatter_chart(x="Inflation rate" , y="GDP" , data=df , color=["#FF0000"])
    
# Use OneHotEncoder to encode categorical features
ct = ColumnTransformer(transformers=[('encode', OneHotEncoder(), ['Target'])], remainder="passthrough")
x = df.iloc[:, :-1]
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
    st.write('Predicted Result Outcome:')
    st.write(predicted_class[0])

from sklearn.metrics import r2_score
st.write('Model Accuracy',r2_score(Y_test,y_pred))