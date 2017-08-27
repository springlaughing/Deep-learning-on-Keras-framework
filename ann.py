# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Converting text data, i.e. country, gender, into numbers (0,1,2)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# As we have 2 categorical independent variables, we create 2 new variables for them
#We encode cathegorical data with fit_transform method
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# In order to scale independent variables, we apply StandardScaler, so that
#our data is in the same range of values
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
# Keras will buid deep learning network using TensorFlow  
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()


# Dense function will ititialize weights, activation function, nodes, etc. 
# We choose activation function at this step.
# For hidden layers: rectifier activation function. It is the best one based on experiments and research.
# For output layer: sigmoid function. Will allow to get probabilities on the output.   
# 11 features gives us 11 independent variables, or 11 input nodes.
# Updating weights is defined by learning rate parameter which decides how weihts are updated. 
# Weights will be randomly initialized by 'uniform' function according to uniform distribution

# Adding the input layer and the first hidden layer. 
# We choose the number of nodes for hidden layer = 6 - average between number of input and output nodes
# However, it could be better tested and optimized. 
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

# Adding the second hidden layer. Our artificial network will have 2 hidden layers.

classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN: applying stocastic gradient descent.
# Optimizer will find optimal set of weghts in NN via 'Adam' - a stocastic gradient descent algorithm. 
# For dependent variable with 2 categories, i.e. for binary outcome, the loss function is 'binary_crossentropy' 
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
# Batch size - number of observation, after which we ajust the weights
# Epoch - round when the whole set is passed through the ANN
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
accuracy = (1518+212)/(1518+212+77+193)
print('Accuracy without additional tuning is', accuracy)
# So, in the y_pred we have predicted which customers will leave (those with probablity of leaving 
# more than 0.5 are marked as True, the others as False)  