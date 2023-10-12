from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
# data = pd.read_csv('PM2.5 Global Air Pollution 2010-2017.csv')
data = pd.read_csv('duke_gpa.csv')

le = LabelEncoder()
le.fit(data['gender'])

# transform the gender column using the LabelEncoder object
data['gender'] = le.transform(data['gender'])

# data = data.drop(['Country Name', 'Country Code'], axis=1) # drop these two because they are non numeric data

# split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.drop('studyweek', axis=1), data['studyweek'], test_size=0.2, random_state=91)
# X_train, X_test, y_train, y_test = train_test_split()

# print the shapes of the training and testing sets
print('Training set shape:', X_train.shape, y_train.shape)
print('Testing set shape:', X_test.shape, y_test.shape)

# create a linear regression model
model = LinearRegression()

# train the model on the training set
model.fit(X_train, y_train)

# make predictions on the testing set
y_pred = model.predict(X_test)

# calculate the mean squared error of the predictions
mse = mean_squared_error(y_test, y_pred)

# calculate the r-squared score of the predictions
r2 = r2_score(y_test, y_pred)

# print the mean squared error
print('Mean squared error:', mse)
print('R2 score:', r2)

# plot the predicted values and the actual values
plt.plot(y_test, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.show()