import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# Sample values for the patient data array
X = np.array([[45, 29, 120, 1, 1, 2, 1, 0],
              [55, 34, 140, 0, 2, 1, 0, 1],
              [30, 26, 100, 1, 3, 3, 1, 0],
              [50, 32, 130, 1, 3, 1, 0, 1],
              [40, 28, 110, 0, 1, 2, 1, 0],
              [38, 25, 115, 1, 2, 3, 0, 1],
              [48, 28, 125, 0, 1, 1, 1, 0],
              [35, 22, 105, 0, 3, 2, 0, 1],
              [60, 35, 150, 1, 1, 1, 1, 0],
              [25, 18, 90, 0, 2, 3, 0, 0],
              [58, 30, 145, 0, 3, 1, 1, 0],
              [50, 32, 130, 1, 2, 2, 0, 1],
              [55, 28, 135, 1, 1, 3, 1, 0],
              [32, 24, 110, 0, 2, 1, 0, 1],
              [45, 27, 120, 1, 3, 3, 0, 0],
              [40, 25, 115, 0, 1, 2, 1, 1],
              [36, 23, 105, 0, 3, 1, 0, 0],
              [48, 30, 130, 1, 1, 3, 0, 1],
              [60, 38, 145, 1, 2, 2, 1, 0],
              [65, 40, 155, 0, 3, 1, 1, 0],
              [51, 35, 135, 0, 1, 3, 0, 1],
              [55, 30, 140, 1, 2, 1, 1, 0],
              [45, 28, 120, 0, 3, 2, 0, 1],
              [37, 25, 110, 1, 1, 3, 1, 0]])

# Sample values for the Y array
Y = np.array([1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# Create the model
model = RandomForestClassifier()

# Train the model
model.fit(X_train, Y_train)

# Test the model
accuracy = model.score(X_test, Y_test)
print("Accuracy:", accuracy)

# Get user input for the X array
name = input("Enter patient name: ")
age = int(input("Enter patient age: "))
BMI = int(input("Enter patient BMI: "))
bp = int(input("Enter patient blood pressure: "))
fh = int(input("Enter patient family history (1 for yes, 0 for no): "))
diet = int(input("Enter patient diet (1 for poor, 2 for average, 3 for good): "))
exercise = int(input("Enter patient exercise (1 for poor, 2 for average, 3 for good): "))
smoking = int(input("Enter patient smoking (1 for yes, 0 for no): "))
alcohol = int(input("Enter patient alcohol consumption (1 for yes, 0 for no): "))

#Create an array for patient data
patient_data = np.array([patient_age, patient_BMI, patient_bp, patient_fh, patient_diet, patient_exercise, patient_smoking, patient_alcohol]).reshape(1, -1)

# Make predictions
prediction = model.predict(patient_data)

# Print the prediction
print("Prediction for", patient_name, ":", prediction)