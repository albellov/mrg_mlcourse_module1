import argparse
import time

from utils import read_mnist, preprocessing_data, save_weights
from my_svm import MySvm
from sklearn.metrics import classification_report


def parse_args():
    path_to_x_train = 'samples/train-images-idx3-ubyte.gz'
    path_to_y_train = 'samples/train-labels-idx1-ubyte.gz'
    path_to_model = 'samples/my_model_multi'

    parser = argparse.ArgumentParser()
    parser.add_argument('-x', '--x_train_dir', default=path_to_x_train,
                        help=f'path to the file with the training sample\'s records, '
                             f'default: {path_to_x_train}')
    parser.add_argument('-y', '--y_train_dir', default=path_to_y_train,
                        help=f'path to the file with the training sample\'s labels, '
                             f'default: {path_to_y_train}')
    parser.add_argument('-m', '--model_output_dir', default=path_to_model,
                        help='path to the file for saving model, '
                             f'default: {path_to_model}')
    parser.add_argument('-v', '--verbosity', default=1,
                        help='is verbosity, default: 1')
    parser.add_argument('-i', '--iter_num', type=int, default=3000,
                        help='number of iterations, default: 3000')
    parser.add_argument('-b', '--batch_size', type=int, default=10,
                        help='mini-batch size, default: 10')
    parser.add_argument('-k', '--kernel', default='poly',
                        help='kernel function: \'linear\' or \'poly\', default: \'poly\'')

    return parser.parse_args()


def main():
    args = parse_args()

    path_to_x_train = args.x_train_dir
    path_to_y_train = args.y_train_dir
    path_to_model = args.model_output_dir
    verbosity = args.verbosity
    iter_num = args.iter_num
    batch_size = args.batch_size

    X_original = read_mnist(path_to_x_train)
    y_original = read_mnist(path_to_y_train)

    X, image_shape = preprocessing_data(X_original)
    y = y_original

    print(f'\nbatch_size: {batch_size}, iter_num: {iter_num}, kernel: {args.kernel}\n')

    X_train, X_val, y_train, y_val = X[:50000], X[50000:], y[:50000], y[50000:]

    clf = MySvm(args.kernel, image_shape=image_shape)
    clf.fit(X_train, y_train, iter_num=iter_num, batch_size=batch_size, verbosity=verbosity)
    prediction_labels = clf.predict(X_val)

    print(classification_report(y_val, prediction_labels, digits=4))
    optimal_weights = clf.get_weights()

    print(f'Saving model to {path_to_model}')
    save_weights(path_to_model, optimal_weights)


if __name__ == "__main__":
    start_time = time.time()
    main()
    exec_time = time.time() - start_time
    print(f'\n\nExecution time: {exec_time//60:5.0f} min, {exec_time%60:1.3} sec\n')
