from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras import models, layers
from keras.callbacks import EarlyStopping

# Load IMDB dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000)

# Pad sequences to a maximum length of 250
X_train_padded = pad_sequences(X_train, maxlen=250)
X_test_padded = pad_sequences(X_test, maxlen=250)

# Concatenate train and test data
data = np.concatenate((X_train_padded, X_test_padded), axis=0)
label = np.concatenate((y_train, y_test), axis=0)

print(X_train_padded.shape)
# The first number, 25000, represents the number of samples or sequences in the array. 
# In this case, it indicates that there are 25,000 sequences in the training set.
# The second number, 250, represents the maximum length of each sequence after padding. 
# In other words, each sequence in the X_train_padded array has been padded (or truncated) to a length of 250.
print(y_train.shape)
print(X_test_padded.shape)
print(y_test.shape)

# Define function for vectorization
def vectorize(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results

# Vectorize data
vectorized_data = vectorize(data)

# Convert labels to float32
label = np.array(label).astype('float32')

# Split data into train and test sets
X_train = vectorized_data[:40000]
y_train = label[:40000]
X_test = vectorized_data[40000:]
y_test = label[40000:]

# Define the neural network model
model = models.Sequential()
model.add(layers.Dense(50, input_shape=(10000,), activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(50, activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(50, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

print(model.summary())

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define early stopping callback
early_stopping = EarlyStopping(monitor='loss', patience=3)

# Train the model
result = model.fit(X_train, y_train,
                   epochs=2,
                   batch_size=500,
                   validation_data=(X_test, y_test),
                   callbacks=[early_stopping])

# Model Accuracy
print(np.mean(result.history['accuracy'])) 

# Make predictions
predictions = model.predict(X_test)
print(predictions)