from BoardGen import *
import csv
import random
import numpy as np

def hide_tiles(matrix, num_hidden):
    hidden_positions = set()
    while len(hidden_positions) < num_hidden:
        row, col = np.random.randint(0, matrix.shape[0]), np.random.randint(0, matrix.shape[1])
        hidden_positions.add((row, col))

    hidden_matrix = np.copy(matrix)
    for row, col in hidden_positions:
        hidden_matrix[row, col] = -1  # Use -1 to represent hidden tiles

    return hidden_matrix

def generate_hidden_3x3(num_hidden):
    num_nines = random.randint(0, 3)
    matrix = np.zeros((3, 3), dtype='int32')

    positions = set()
    while len(positions) < num_nines:
        row, col = np.random.randint(0, 3, size=2)
        positions.add((row, col))

    for row, col in positions:
        matrix[row, col] = 9

    labels = 0
    for row, col in positions:
        labels[row, col] = 1  # 1 indicates the presence of a mine

    matrix = add_one_to_nines(matrix)

    # Determine the number of non-mine tiles to hide
    num_non_mine_hidden = num_hidden - num_nines

    # Hide both mines and non-mine tiles
    hidden_matrix = hide_tiles(matrix, num_hidden)

    return hidden_matrix.flatten(), labels.flatten()

def generate_training_data(num_samples, num_hidden):
    csv_file_path = "input/training_data_hidden_2d.csv"

    with open(csv_file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)

        csv_writer.writerow(['Col1', 'Col2', 'Col3', 'Col4', 'Col5', 'Col6', 'Col7', 'Col8', 'Col9','Col10'])

        for j in range(1, num_hidden):
            for i in range(num_samples // num_hidden):
                data, labels = generate_hidden_3x3(j)
                combined_data = np.concatenate((data, labels))  # Combine input and labels
                csv_writer.writerow(combined_data)

def generate_validating_data(num_samples, num_hidden):
    csv_file_path = "input/validating_data_hidden_2d.csv"

    with open(csv_file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)

        csv_writer.writerow(['Col1', 'Col2', 'Col3', 'Col4', 'Col5', 'Col6', 'Col7', 'Col8', 'Col9','Col10'])

        for j in range(num_hidden):
            for i in range(10):
                data, labels = generate_hidden_3x3(j)
                combined_data = np.concatenate((data, labels))  # Combine input and labels
                csv_writer.writerow(combined_data)

def main():
    num_samples = 100
    num_hidden = 4  # Adjust the total number of hidden tiles as needed
    generate_training_data(num_samples, num_hidden)
    generate_validating_data(num_samples/10, num_hidden)

if __name__ == '__main__':
    main()
