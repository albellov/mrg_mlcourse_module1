import numpy as np
import argparse

from utils import load_weights, read_mnist, preprocessing_data
from sklearn.metrics import classification_report


def predict(X, weights):
    pred = np.dot(X, weights.T).T
    return np.argmax(pred, axis=0)


def prediction_by_file(X, y, filename, digits=2):
    weights = load_weights(filename)
    prediction_by_weights(X, y, weights, digits=digits)


def prediction_by_weights(X, y, weights, digits=2):
    predicted_labels = predict(X, weights)
    print(classification_report(y, predicted_labels, digits=digits))


def accuracy(X, y, weights):
    pred = predict(X, weights)
    return np.mean(pred == y)


def parse_args():
    path_to_x_test = 'samples/t10k-images-idx3-ubyte.gz'
    path_to_y_test = 'samples/t10k-labels-idx1-ubyte.gz'
    path_to_model = 'samples/my_model'

    parser = argparse.ArgumentParser()
    parser.add_argument('-x', '--x_test_dir', default=path_to_x_test,
                        help=f'путь к файлу, в котором лежат рекорды тестовой выборки, по умолчанию: {path_to_x_test}')
    parser.add_argument('-y', '--y_test_dir', default=path_to_y_test,
                        help=f'путь к файлу, в котором лежат метки тестовой выборки, по умолчанию: {path_to_y_test}')
    parser.add_argument('-m', '--model_input_dir', default=path_to_model,
                        help='путь к файлу, в который скрипт сохраняет обученную модель, '
                             f'по умолчанию: {path_to_model}')

    return parser.parse_args()


def main():

    args = parse_args()

    X_original = read_mnist(args.x_test_dir)
    X_test = preprocessing_data(X_original)
    y_test = read_mnist(args.y_test_dir)

    print('Metrics on the test data:\n')
    prediction_by_file(X_test, y_test, args.model_input_dir, 4)


if __name__ == "__main__":
    main()
