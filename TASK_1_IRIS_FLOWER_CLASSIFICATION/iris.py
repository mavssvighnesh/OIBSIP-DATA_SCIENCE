# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Iris dataset
iris_data = pd.read_csv('V:/Iris.csv')

# Displaying basic information about the dataset
iris_data.info()  # Show information about columns and their data types
iris_data.head()  # Display the first few rows of the dataset
iris_data.tail()  # Display the last few rows of the dataset
iris_data.describe()  # Statistical summary of numerical columns

# Checking for missing values
iris_data.isnull()  # Check for missing values in the dataset
iris_data.isnull().sum()  # Summarize missing values in each column

# Count the occurrences of each species
species_count = iris_data['Species'].value_counts()
print("All individual species column of the dataset: ")
print(species_count)

# Visualizing Sepal Length, Sepal Width, and Petal Width by species
plt.bar(iris_data['Species'], iris_data['SepalLengthCm'])
plt.title("Sepal length VS Species")
plt.xlabel("Name of the Species")
plt.ylabel("Sepal Length")
plt.grid(True)
plt.show()

#graph between the sepal width and species they are 
plt.bar(iris_data['Species'], iris_data['SepalWidthCm'])
plt.title("Sepal Width VS Species")
plt.xlabel("Name of the Species")
plt.ylabel("Sepal Width")
plt.grid(True)
plt.show()

#Graph between the petal width and the species they are 
plt.bar(iris_data['Species'], iris_data['PetalWidthCm'])
plt.title("Petal Width VS Species")
plt.xlabel("Name of the Species")
plt.ylabel("Petal Width")
plt.grid(True)
plt.show()

# Visualizing pairplots to understand relationships among features
plt.figure(figsize=(6, 6))
sns.pairplot(iris_data, hue='Species')
plt.show()

# Split the data into features (X) and labels (y)
X = iris_data.drop(['Id', 'Species'], axis=1)
y = iris_data['Species']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a Random Forest Classifier model
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rfc.predict(X_test)

# Calculate and display the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Visualize the confusion matrix
cm = confusion_matrix(y_pred, y_test)
sns.heatmap(cm, annot=True)
plt.show()

# Visualize the actual species based on Sepal Length and Sepal Width
species_mapping = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
plt.scatter(X_test['SepalLengthCm'], X_test['SepalWidthCm'], c=y_test.map(species_mapping), cmap='viridis')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('Actual Iris Species')
plt.show()

# Visualize the predicted species based on Sepal Length and Sepal Width
plt.scatter(X_test['SepalLengthCm'], X_test['SepalWidthCm'], c=pd.Series(y_pred).map(species_mapping), cmap='viridis')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('Predicted Iris Species')
plt.show()