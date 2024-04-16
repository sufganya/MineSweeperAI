import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten , BatchNormalization , LeakyReLU
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
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

    # Flatten the input
    model.add(Flatten(input_shape=(input_shape,)))

    # Dense layers with ReLU activation

    model.add(Dense(256, activation='relu'))

    model.add(Dense(128, activation='relu'))

    model.add(Dense(64, activation='relu'))

    # Output layer with sigmoid activation for binary classification
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    initial_lr = 0.0001
    loss = 'binary_crossentropy'
    model.compile(Adam(learning_rate=initial_lr), loss=loss, metrics=['accuracy'])

    return model

# Plot training history
def plot_training_history(history):
    plt.plot(history.history['accuracy'], label='train_accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.legend()
    plt.show()

    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend()
    plt.show()

# Main function
def main():
    csv_file_path = "input/training_adv_2.csv"  # Replace with your actual file path
    train_features, val_features, train_labels, val_labels = load_and_preprocess_data(csv_file_path)

    model = build_model(train_features.shape[1])

    # Use early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train the model
    epochs = 200  # Adjust as needed
    batch_size = 32  # Adjust as needed
    history = model.fit(
        train_features, train_labels,
        batch_size=batch_size, epochs=epochs,
        validation_data=(val_features, val_labels),
        callbacks=[early_stopping]
    )

    # Save or visualize the training history
    plot_training_history(history)

    # Save the model
    model.save('output/minesweeper_AI_Dense_binary_03.h5')  # Replace with your desired model path

    # Save or visualize the training history
    plot_training_history(history)

    model = models.load_model("output/minesweeper_AI_Dense_binary_02.h5")

    new_data = np.array([[2,3,2,-1,-1,-1,-1,-1,-1]])
    predictions = model.predict(new_data)
    print(predictions)
    new_data = np.array([[1, 2, 1, -1, -1, -1, -1, -1, -1]])
    predictions = model.predict(new_data)
    print(predictions)
    new_data = np.array([[1, 1, 1, -1, 11, -1, -1, -1, -1]])
    predictions = model.predict(new_data)
    print(predictions)
    new_data = np.array([[3, 3, 3, -1, -1, -1, -1, -1, -1]])
    predictions = model.predict(new_data)
    print(predictions)

if __name__ == '__main__':
    main()
