import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Reshape, BatchNormalization, Dropout, LeakyReLU
from keras.optimizers import Adam, RMSprop, SGD
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from keras.preprocessing.image import ImageDataGenerator
from kerastuner import RandomSearch
from keras import models
from kerastuner.engine.hyperparameters import HyperParameters

# Load and preprocess data
def load_and_preprocess_data(csv_file_path):
    df = pd.read_csv(csv_file_path)

    # Separate features and labels
    features = df.iloc[:, 1:].values  # Assuming the first column is the label
    labels = df.iloc[:, 0].values  # Assuming the first column is the label

    # Train-test split
    train_features, val_features, train_labels, val_labels = train_test_split(features, labels, test_size=0.1)

    return train_features, val_features, train_labels, val_labels

# Build the model
def build_model():
    model = Sequential()

    # Flatten the input
    model.add(Reshape((5, 5, 1), input_shape=(25, 1)))

    # First convolutional layer
    model.add(Conv2D(64, kernel_size=3, padding='same', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.2))
    # Second convolutional layer
    model.add(Conv2D(80, kernel_size=3, padding='same', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.2))
    # Flatten the output
    model.add(Flatten())

    # First dense layer
    model.add(Dense(96, kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.2))
    # Second dense layer
    model.add(Dense(64, kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.2))

    # Output layer
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'])

    return model

# Main function
def main():
    csv_file_path = "input/training_newgen.csv"  # Replace with your actual file path
    train_features, val_features, train_labels, val_labels = load_and_preprocess_data(csv_file_path)

    model = build_model()

    # Use early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train the model
    epochs = 200  # Adjust as needed
    batch_size = 64  # Adjust as needed
    history = model.fit(
        train_features, train_labels,
        batch_size=batch_size, epochs=epochs,
        validation_data=(val_features, val_labels),
        callbacks=[early_stopping]
    )

    # Calculate AUC-ROC for validation set
    val_pred = model.predict(val_features)

    prob_true, prob_pred = calibration_curve(val_labels, val_pred, n_bins=10, strategy='uniform')

    # Plot reliability diagram
    plt.plot(prob_pred, prob_true, marker='o', linestyle='-')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # Perfect calibration line
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Observed probability')
    plt.title('Reliability Diagram')
    plt.show()

    # Save the model
    model.save('output/minesweeper_AI_Conv2D_binary_Adam.h5')  # Replace with your desired model path

    # Save or visualize the training history
    plot_training_history(history)

# Plot training history
def plot_training_history(history):
    plt.plot(history.history['binary_accuracy'], label='train_accuracy')
    plt.plot(history.history['val_binary_accuracy'], label='val_accuracy')
    plt.legend()
    plt.show()

    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend()
    plt.show()

def testing():
    # Load the trained model
    model = models.load_model("output/minesweeper_AI_Conv2D_binary_Adam.h5")

    # Load the testing dataset
    testing_data = pd.read_csv("input/testing_newgen.csv")

    # Separate features (X_test) and labels (y_test)
    X_test = testing_data.iloc[:, 1:].values  # Assuming the first column is not the label
    y_test = testing_data.iloc[:, 0].values   # Assuming the first column is the label

    # Evaluate the model on the entire testing dataset
    loss, accuracy = model.evaluate(X_test, y_test, batch_size=1)

    print("Testing Loss:", loss)
    print("Testing Accuracy:", accuracy)


if __name__ == '__main__':
    testing()
