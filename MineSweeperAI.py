import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,brier_score_loss,auc,roc_curve, precision_recall_curve, f1_score, log_loss
from sklearn.calibration import calibration_curve
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten , BatchNormalization , LeakyReLU
from keras.optimizers import Adam, SGD, Adamax, RMSprop,Adagrad
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from keras import models
import matplotlib.pyplot as plt
from scipy.ndimage import rotate

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
def build_model(input_shape, dropout_rate=0.2, l2_penalty=0.001):
    model = Sequential()

    # Flatten the input
    model.add(Flatten(input_shape=(input_shape,)))

    # # Dense layers with ReLU activation and Batch Normalization
    # model.add(Dense(256, kernel_regularizer=l2(l2_penalty)))
    # model.add(BatchNormalization())
    # model.add(LeakyReLU(alpha=0.1))
    # model.add(Dropout(dropout_rate))
    #
    # model.add(Dense(128, kernel_regularizer=l2(l2_penalty)))
    # model.add(BatchNormalization())
    # model.add(LeakyReLU(alpha=0.1))
    # model.add(Dropout(dropout_rate))

    model.add(Dense(64, kernel_regularizer=l2(l2_penalty)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(dropout_rate))

    model.add(Dense(32, kernel_regularizer=l2(l2_penalty)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(dropout_rate))

    # Output layer with sigmoid activation for binary classification
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    #initial_lr = 0.0001
    optimizer = Adam()
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

    model = build_model(train_features.shape[1])

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

    # Save the model
    model.save('output/minesweeper_AI_Dense_binary_Adam_Cheap.h5')  # Replace with your desired model path

    # Save or visualize the training history
    plot_training_history(history)

    # Compute calibration curve
    prob_true, prob_pred = calibration_curve(val_labels, val_pred, n_bins=10, strategy='uniform')

    # Plot reliability diagram
    plt.plot(prob_pred, prob_true, marker='o', linestyle='-')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # Perfect calibration line
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Observed probability')
    plt.title('Reliability Diagram')
    plt.show()

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(val_labels, val_pred)

    # Compute AUC-ROC
    auc_roc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % auc_roc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()

    # Print evaluation metrics
    auc_roc = roc_auc_score(val_labels, val_pred)
    print("AUC-ROC:", auc_roc)

    brier_score = brier_score_loss(val_labels, val_pred)
    print("Brier Score:", brier_score)

    precision, recall, _ = precision_recall_curve(val_labels, val_pred)
    auc_pr = auc(recall, precision)
    print("AUC-PR:", auc_pr)

    f1 = f1_score(val_labels, (val_pred >= 0.5).astype(int))
    print("F1 Score:", f1)

    logloss = log_loss(val_labels, val_pred)
    print("Log Loss:", logloss)

def testing():
    # Load the trained model
    model = models.load_model("output/minesweeper_AI_Dense_binary_Adam_Cheap.h5")

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
    main()
