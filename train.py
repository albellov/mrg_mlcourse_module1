import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from utils import read_mnist, preprocessing_data, save_weights
from predict import prediction_by_weights


def _hinge_loss_dataset(predicted_values, true_values):
    res = 1 - predicted_values * true_values
    res[np.where(res < 0)] = 0
    return res.mean()


def hinge_loss_in_point(X, y, weights):
    predicted_values = np.dot(X, weights)
    return _hinge_loss_dataset(predicted_values, y)


def regul_loss_in_point(my_loss_fun, coef, ord_=2):
    def loss_fun(X, y, weights):
        return my_loss_fun(X, y, weights) + coef * np.linalg.norm(weights[1:], ord=ord_)

    return loss_fun


def _init_weights(features_num, a, b):
    return np.random.uniform(a, b, features_num)


def _num_gradient(f, X, y, model_weights, w_delta=1e-3):
    dim = len(model_weights)
    cur_f = f(X, y, model_weights)

    diff = lambda dw: f(X, y, model_weights + dw) - cur_f
    delta_matrix = np.eye(dim) * w_delta

    grad = np.array([diff(dx) for dx in delta_matrix]) / w_delta

    return grad


def SGD(loss_fun, X, y, model_weights, learning_rate, iter_num=2000, elems=10, decr=1.01):
    losses = []
    for i in range(iter_num):
        indexes = np.random.choice(range(len(X)), elems)
        grad = _num_gradient(loss_fun, X[indexes], y[indexes], model_weights)
        model_weights -= learning_rate * grad / np.linalg.norm(grad)

        cur_loss = loss_fun(X, y, model_weights)
        losses.append(cur_loss)

        learning_rate = max(5e-3, learning_rate / decr)
        if i % 100 == 0:
            print(f'\titer_num: {i},\tlearning_rate: {learning_rate},\tloss_in_point: {cur_loss}')
            # print(f'\tgrad : {np.linalg.norm(grad)}')

    return model_weights, losses


def _class_balancing(X, y):
    indexes = [np.where(y == -1), np.where(y == 1)]
    counts = [len(indexes[0][0]), len(indexes[1][0])]

    if counts[0] > counts[1]:
        num = int(round(counts[0] / counts[1]))
        ind = indexes[1]
    else:
        num = int(round(counts[1] / counts[0]))
        ind = indexes[0]

    X_original = X.copy()
    y_original = y.copy()

    for _ in range(num - 1):
        X = np.vstack((X, X_original[ind, :][0]))
        y = np.concatenate((y, y_original[ind]))

    return X, y


def fit_model(loss_fun, X_, y_, learning_rate=1,
              iter_num=2000,
              elems=50, decr=1.001):
    optimal_weights = []
    losses = []
    y = y_ + 2

    for label in np.unique(y):
        print(f'Training for label = {label-2}')
        yy = y.copy()

        yy[np.where(yy != label)] = -1
        yy[np.where(yy == label)] = 1

        X, yy = _class_balancing(X_, yy)
        _, features_num = X.shape
        init_w = _init_weights(features_num, -0.5, 0.5)

        optimal_weight, loss = SGD(loss_fun, X, yy, init_w, learning_rate, iter_num, elems, decr)

        optimal_weights.append(optimal_weight)
        losses.append(loss)

    return np.array(optimal_weights), np.array(losses)


def main():
    path_to_x_train = 'samples/train-images-idx3-ubyte.gz'
    path_to_y_train = 'samples/train-labels-idx1-ubyte.gz'
    path_to_model = 'samples/my_model'

    parser = argparse.ArgumentParser()
    parser.add_argument('-x', '--x_train_dir', default=path_to_x_train,
                        help=f'путь к файлу, в котором лежат рекорды обучающей выборки, по умолчанию: {path_to_x_train}')
    parser.add_argument('-y', '--y_train_dir', default=path_to_y_train,
                        help=f'путь к файлу, в котором лежат метки обучающей выборки, по умолчанию: {path_to_y_train}')
    parser.add_argument('-m', '--model_output_dir', default=path_to_model,
                        help='путь к файлу, в который скрипт сохраняет обученную модель')
    parser.add_argument('-v', '--verbosity', default=0,
                        help='отображение хода обучения, по умолчанию: 0')

    args = parser.parse_args()

    X_original = read_mnist(args.x_train_dir)
    y_original = read_mnist(args.y_train_dir)

    X = preprocessing_data(X_original)
    y = y_original.astype(np.int8)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    coef = 5000

    print(f'C = {coef}\n')
    regul_loss = regul_loss_in_point(hinge_loss_in_point, 0.5 / coef, ord_=1)
    optimal_weights, _ = fit_model(regul_loss, X_train, y_train,
                                   learning_rate=1, iter_num=5001, elems=10, decr=1.0001
                                   )

    print(f'Saving model to {args.model_output_dir}')
    save_weights(args.model_output_dir, optimal_weights)

    print('Metrics on the train data:\n')
    prediction_by_weights(X_train, y_train, optimal_weights, 4)


if __name__ == "__main__":
    main()
