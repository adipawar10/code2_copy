import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Loading the dataset.
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
iris_df = pd.read_csv("iris-species.csv")

# Adding a column in the Iris DataFrame to resemble the non-numeric 'Species' column as numeric using the 'map()' function.
# Creating the numeric target column 'Label' to 'iris_df' using the 'map()' function.
iris_df['Label'] = iris_df['Species'].map({'Iris-setosa': 0, 'Iris-virginica': 1, 'Iris-versicolor':2})

# Creating a model for Support Vector classification to classify the flower types into labels '0', '1', and '2'.

# Creating features and target DataFrames.
X = iris_df[['SepalLengthCm','SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = iris_df['Label']

# Splitting the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

# Creating the SVC model and storing the accuracy score in a variable 'score'.
svc_model = SVC(kernel = 'linear')
svc_model.fit(X_train, y_train)
score = svc_model.score(X_train, y_train)

@st.cache()
def prediction(SepalLength, SepalWidth, PetalLength, PetalWidth, model):
	species=model.predict([[SepalLength, SepalWidth, PetalLength, PetalWidth]])
	if species[0]==0:
	    return "Iris-setosa"
	elif species[0]==1:
	    return "Iris-virginica"
	else:
	    return "Iris-versicolor"
obj1=RandomForestClassifier(n_estimators=100,n_jobs=-1)
obj1.fit(X_train,y_train)
score_1=obj1.score(X_train,y_train)
obj2=LogisticRegression()
obj2.fit(X_train,y_train)
score2=obj2.score(X_train,y_train)
st.title("Iris Flower Species Prediction")
sl=st.sidebar.slider("SepalLength",float(iris_df["SepalLengthCm"].min()),float(iris_df["SepalLengthCm"].max()))
sw=st.sidebar.slider("SepalWIdth",float(iris_df["SepalWidthCm"].min()),float(iris_df["SepalWidthCm"].max()))
pl=st.sidebar.slider("PetalLength",float(iris_df["PetalLengthCm"].min()),float(iris_df["PetalLengthCm"].max()))
pw=st.sidebar.slider("PetalWidth",float(iris_df["PetalWidthCm"].min()),float(iris_df["PetalWidthCm"].max()))
classification_selector=st.sidebar.selectbox("Classification Selector",["Support Vector Machine","Logisitc Regression","RandomForestClassifier"])
if st.button("Predict"):
    if classification_selector == "Support Vector Machine" :
         pred_val = prediction(sl,sw,pl,pw,svc_model)
         st.write("Accuracy",score)
         st.write("Species Preadicted:",pred_val)

    elif classification_selector == "Logistic Regression" :
         pred_val = prediction(sl,sw,pl,pw,obj2)
         st.write("Accuracy",score2)
         st.write("Species Preadicted:",pred_val)


    else:
    	pred_val = prediction(sl,sw,pl,pw,obj1)
    	st.write("Accuracy",score_1)
    	st.write("Species Preadicted:",pred_val)



