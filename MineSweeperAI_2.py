import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv1D, GlobalAveragePooling1D
from keras.optimizers import Adam
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

    model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=(input_shape, 1)))
    model.add(Conv1D(64, kernel_size=3, activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    initial_lr = 0.0001
    loss = 'binary_crossentropy'
    model.compile(Adam(lr=initial_lr), loss=loss, metrics=['accuracy'])

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
    # csv_file_path = "input/training.csv"  # Replace with your actual file path
    # train_features, val_features, train_labels, val_labels = load_and_preprocess_data(csv_file_path)
    #
    # # Expand dimensions for compatibility with Conv1D
    # train_features = np.expand_dims(train_features, axis=-1)
    # val_features = np.expand_dims(val_features, axis=-1)
    #
    # model = build_model(train_features.shape[1])
    #
    # epochs = 50
    # batch_size = 32
    # history = model.fit(
    #     train_features, train_labels,
    #     batch_size=batch_size, epochs=epochs,
    #     validation_data=(val_features, val_labels)
    # )
    #
    # # Save or visualize the training history
    # plot_training_history(history)

    # Save the model
    # model.save('output/minesweeper_AI_Conv1D_binary.h5')

    model = models.load_model("output/minesweeper_AI_Conv1D_binary.h5")

    new_data = np.array([[2,3,2,-1,-1,-1,-1,-1,-1]])
    predictions = model.predict(new_data)
    print(predictions)

    new_data2 = np.array([[1, 2, 1, -1, -1, -1, -1, -1, -1]])
    predictions = model.predict(new_data2)
    print(predictions)

if __name__ == '__main__':
    main()
