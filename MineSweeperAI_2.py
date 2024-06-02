import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.calibration import calibration_curve
from keras import models
import matplotlib.pyplot as plt

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
def build_model(input_shape):
    model = Sequential()

    # Assuming train_features.shape[1] is the length of your input array
    input_shape = (input_shape, 1)

    # Convolutional layers
    model.add(Conv1D(32, 3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(MaxPooling1D(2))

    # Flatten the output of the convolutional layers
    model.add(Flatten())

    # Dense layers
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))

    # Output layer with sigmoid activation for binary classification
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    initial_lr = 0.0001
    optimizer = Adam(learning_rate=initial_lr)
    loss = 'binary_crossentropy'
    model.compile(optimizer, loss=loss, metrics=['binary_accuracy'])

    return model


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

# Main function
def main():
    csv_file_path = "input/training_newgen.csv"  # Replace with your actual file path
    train_features, val_features, train_labels, val_labels = load_and_preprocess_data(csv_file_path)

    # Expand dimensions for compatibility with Conv1D
    train_features = np.expand_dims(train_features, axis=-1)
    val_features = np.expand_dims(val_features, axis=-1)

    model = build_model(train_features.shape[1])

    # Use early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    epochs = 200
    batch_size = 64
    history = model.fit(
        train_features, train_labels,
        batch_size=batch_size, epochs=epochs,
        validation_data=(val_features, val_labels),
        callbacks=[early_stopping]
    )

    # Save or visualize the training history
    plot_training_history(history)

    val_pred = model.predict(val_features)

    prob_true, prob_pred = calibration_curve(val_labels, val_pred, n_bins=10, strategy='uniform')

    # Plot reliability diagram
    plt.plot(prob_pred, prob_true, marker='o', linestyle='-')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # Perfect calibration line
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Observed probability')
    plt.title('Reliability Diagram')
    plt.show()

    #Save the model
    model.save('output/minesweeper_AI_Conv1D_binary_Adam.h5')


if __name__ == '__main__':
    main()
