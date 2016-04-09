from tensor_dynamic.data_functions import normalize
import numpy as np

INPUT_COLUMNS = ['']


def num_to_one_hot(char):
    if char == 'A':
        return [1.0, 0.0, 0.0, 0.0, 0.0]
    elif char == 'B':
        return [0.0, 1.0, 0.0, 0.0, 0.0]
    elif char == 'C':
        return [0.0, 0.0, 1.0, 0.0, 0.0]
    elif char == 'D':
        return [0.0, 0.0, 0.0, 1.0, 0.0]
    else:
        return [0.0, 0.0, 0.0, 0.0, 1.0]


def get_data(data_dir):
    inputs = []
    outputs = []
    with open(data_dir + 'servo.data') as f:
        for line in f:
            items = line.split(' ')
            input_cols = items[0].split(',')[:-1]
            inputs.append(num_to_one_hot(input_cols[0])+num_to_one_hot(input_cols[1])+[float(input_cols[2]),float(input_cols[2])])
            outputs.append([float(items[1])])
    return np.array(normalize(inputs), dtype=np.float64), np.array(normalize(outputs), dtype=np.float64)


if __name__ == '__main__':
    x, y = get_data('')
    print x
    print y
