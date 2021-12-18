# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm 

# Import data
spam = pd.read_csv('spam_data.csv')

# Build the model
# Train dataset: used to fit the machine learning model
# Test dataset: used to evaluate the fit of the machine learning model
z = spam['Message']
y = spam["Category"]
z_train, z_test,y_train, y_test = train_test_split(z,y,test_size = 0.2)
# test_size=0.2 -> sets the testing set to 20 percent of z and y

# It is a text, so we transform it into numbers that correspond to words and we also have the frequency of each word.
# Info -> https://es.acervolima.com/2021/02/09/uso-de-countvectorizer-para-extraer-caracteristicas-de-texto/
cv = CountVectorizer()
features_train = cv.fit_transform(z_train)

# Building the model
# SVM ->  the support vector machine algorithm, is a linear model for classification and regression
model = svm.SVC()
model.fit(features_train,y_train)

# Testing our email spam detector
features_test = cv.transform(z_test)
print("Accuracy:",model.score(features_test,y_test))
