import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras import models, layers
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Read data
df = pd.read_csv("HousingData.csv")

# Separate features and target variable
X = df.iloc[:, :-1].values  # Assuming the last column is the target variable
y = df.iloc[:, -1].values

# Standardize features
sc = StandardScaler()
X = sc.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the neural network model
model = models.Sequential()
model.add(layers.Dense(62, input_dim=13, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1))

print(model.summary())

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

# Define early stopping criteria
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

# Train the model
result = model.fit(X_train, y_train,
                   validation_split=0.2,
                   epochs=100,
                   batch_size=32,
                   callbacks=[early_stopping])

# Plot loss over epochs
plt.plot(result.history['loss'])
plt.plot(result.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Training', 'Validation'])
plt.show()

# Evaluate the model on the test set
mae = model.evaluate(X_test, y_test)
print(mae)
