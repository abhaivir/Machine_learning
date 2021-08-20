# Machine_learning
Cancer Melignant vs benign tumour diagnosis using Keras
import pandas as pd
# using source data from google and tested on google : colab.research.google.com
dataset = pd.read_csv('/content/cancer.csv')
# removing diagnosis column from dataset to keep it aside in y variable
x= dataset.drop(columns= "diagnosis(1=m, 0=b)")
x.head()
# seperating part of data for checking
y = dataset["diagnosis(1=m, 0=b)"]
y.head()
# Using Scikit leanr for spiltting data between train, test 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2)
# test_size represent the ratio in which test varaible got data from dataset

# using tensorflow with keras
import tensorflow as tf
model = tf.keras.models.Sequential() 
# using neural network for analysis
# creating layes of input, dense and final in network with 256 units being output at that layer,
# with activation as 'sigmoid' to generate value between 0,1 and to predict the output value,
# probability of anything is between 0 to 1: so, sigmoid is the right choice.
model.add(tf.keras.layers.Dense(256, input_shape = x_train.shape,activation='sigmoid'))
model.add(tf.keras.layers.Dense(256, activation='sigmoid'))
model.add(tf.keras.layers.Dense(1,activation='sigmoid'))

# compiling model with accuracy and optimiser
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
# here the epochs are the count of iterations the model gets the data for familarity
model.fit(x_train,y_train,epochs=600)

# At last, evaluation process on the model.
model.evaluate(x_test,y_test)

## Output: 
# loss: 0.0700 - accuracy: 0.9649
# [0.07001126557588577, 0.9649122953414917]
# Achieved an accuracy of 96% on the given small dataset.


# Enjoy :)
