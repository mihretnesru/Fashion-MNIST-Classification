# Import neccesary libraries
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier
from pyspark.sql import SparkSession
from sklearn.model_selection import train_test_split

# Initialize Spark session
spark = SparkSession.builder \
    .appName("FashionMNIST-Model-Training") \
    .getOrCreate()

# Load the Fashion MNIST dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Preprocess the data: normalize and reshape
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255

# Convert labels to categorical (one-hot encoding)
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Function to build the CNN model with hyperparameters for tuning
def build_model(learning_rate=0.001, dropout_rate=0.3):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(dropout_rate),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# Wrap the Keras model using KerasClassifier
model = KerasClassifier(model=build_model, epochs=10, verbose=0)

# Define parameter grid for hyperparameter tuning
param_grid = {
    'model__learning_rate': [0.001, 0.01],   # Learning rate for optimizer
    'model__dropout_rate': [0.3, 0.5],        # Dropout rate in the model
    'batch_size': [32, 64]                    # Batch size
}

# Set up the grid search with cross-validation
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, verbose=1)

# Perform the grid search on the training data 
grid_search.fit(x_train, y_train)

# Print best parameters and scores
print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best accuracy: {grid_search.best_score_}")

# Use the best model from grid search
best_model = grid_search.best_estimator_

# Access the underlying Keras model 
keras_model = best_model.model  
if callable(keras_model):
    keras_model = keras_model()  
   
print("Model Architecture:")
keras_model.summary()

# Train the best model
history = keras_model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=grid_search.best_params_['batch_size'],
    validation_data=(x_test, y_test),
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)]
)

# Evaluate the best model on the test data
test_loss, test_acc = keras_model.evaluate(x_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc}, Test loss: {test_loss}")

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Model Accuracy')
plt.legend(loc='upper left')
plt.show()

# Plot training and validation loss
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Model Loss')
plt.legend(loc='upper left')
plt.show()

# Stop Spark session 
spark.stop()
