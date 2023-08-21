import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

from common import cosine_similarity_loss

# Model definition


def upsampling_model():
    input_layer = layers.Input(shape=(32,))
    x = layers.Dense(256, activation='relu')(input_layer)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(1024, activation='relu')(x)
    output_layer = layers.Dense(1536, activation='linear')(x)
    model = models.Model(inputs=input_layer, outputs=output_layer)
    return model


# Define the model
model = upsampling_model()

# Load the embeddings and the cosine similarity matrix from .npy files
X_train = np.load('./projected_train_embs.npy')
cosine_angle_similarity_matrix = np.load('./og_train_cos_theta.npy')
# Create an array of indices corresponding to each embedding in X_train
y_train_indices = np.arange(len(X_train))

# Print the shapes
print("Shape of X_train:", X_train.shape)
print("Shape of cosine_angle_similarity_matrix:",
      cosine_angle_similarity_matrix.shape)
print("Shape of y_train_indices:", y_train_indices.shape)

# Compile the model with the custom loss function
model.compile(optimizer='adam', loss=cosine_similarity_loss(
    cosine_angle_similarity_matrix))

# Create an early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)

# Modify your fit function to include the early stopping callback
history = model.fit(X_train, y_train_indices, epochs=100,
                    batch_size=30, validation_split=0.2, callbacks=[early_stopping])

# Save the trained model
model.save("./trained")
