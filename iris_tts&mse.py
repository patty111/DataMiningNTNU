from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import time
start = time.time()

data = []

# Load the iris dataset
iris = load_iris()
for i in range(1, 100):
    print(i)
    hist = 0
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=(i/100)) # use 0.2 for better accuracy
    
    # Define a decision tree classifier
    classifier = DecisionTreeClassifier()
    for itr in range(1000):
        # Train the classifier on the training set
        classifier.fit(X_train, y_train)

        # Predict the classes of the testing set
        y_pred = classifier.predict(X_test)

        # Calculate the accuracy of the classifier on the testing set
        accuracy = accuracy_score(y_test, y_pred)
        hist += accuracy

    # print('Accuracy:', accuracy)
    data.append(hist/1000)

print(time.time() - start)
plt.plot(data)
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Line Plot of Data')
plt.show()