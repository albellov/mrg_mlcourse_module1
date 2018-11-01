import argparse
import time

from utils import load_weights, read_mnist, preprocessing_data
from sklearn.metrics import classification_report
from my_svm import MySvm


def parse_args():
    path_to_x_test = 'samples/t10k-images-idx3-ubyte.gz'
    path_to_y_test = 'samples/t10k-labels-idx1-ubyte.gz'
    path_to_model = 'samples/my_model'

    parser = argparse.ArgumentParser()
    parser.add_argument('-x', '--x_test_dir', default=path_to_x_test,
                        help=f'path to the file with the testing sample\'s records, '
                             f'default: {path_to_x_test}')
    parser.add_argument('-y', '--y_test_dir', default=path_to_y_test,
                        help=f'path to the file with the testing sample\'s labels, '
                             f'default: {path_to_y_test}')
    parser.add_argument('-m', '--model_input_dir', default=path_to_model,
                        help='path to the file for loading model, '
                             f'default: {path_to_model}')
    parser.add_argument('-k', '--kernel', default='poly',
                        help='kernel function: \'linear\' or \'poly\', default: \'poly\'')
    return parser.parse_args()


def main():

    args = parse_args()
    path_to_x_test = args.x_test_dir
    path_to_y_test = args.y_test_dir
    path_to_model = args.model_input_dir
    kernel = args.kernel

    X_original = read_mnist(path_to_x_test)
    X_test, image_shape = preprocessing_data(X_original)
    y_test = read_mnist(path_to_y_test)

    weights = load_weights(path_to_model)

    clf = MySvm(kernel_type=kernel, image_shape=image_shape)
    clf.load_weights(weights)
    predict_labels = clf.predict(X_test)

    print('Metrics on the test data:\n')
    print(classification_report(y_test, predict_labels, digits=4))


if __name__ == "__main__":
    start_time = time.time()
    main()
    exec_time = time.time() - start_time
    print(f'\n\nExecution time: {exec_time//60:5.0f} min, {exec_time%60:1.3} sec\n')
