import numpy as np
import argparse

from predict import prediction_by_weights, accuracy
from sklearn.model_selection import train_test_split, KFold
from utils import read_mnist, preprocessing_data, save_weights


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

    def diff(dw):
        return f(X, y, model_weights + dw) - cur_f

    delta_matrix = np.eye(dim) * w_delta

    grad = np.array([diff(dx) for dx in delta_matrix]) / w_delta

    return grad


def SGD(loss_fun, X, y, model_weights, learning_rate, iter_num=2000, elems=10,
        decr=1.01, min_learning_rate=1e-3, verbosity=1):
    losses = []
    for i in range(iter_num):
        indexes = np.random.choice(range(len(X)), elems)
        grad = _num_gradient(loss_fun, X[indexes], y[indexes], model_weights)
        model_weights -= learning_rate * grad / np.linalg.norm(grad)

        cur_loss = loss_fun(X, y, model_weights)
        losses.append(cur_loss)

        learning_rate = max(min_learning_rate, learning_rate / decr)
        if verbosity and (i % 100 == 0 or i == iter_num-1):
            print(f'\titer_num: {i},\tlearning_rate: {learning_rate:2.3f},\tloss_in_point: {cur_loss:2.3f}')

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

    # for _ in range(num - 1):
    for _ in range(num):
        X = np.vstack((X, X_original[ind, :][0]))
        y = np.concatenate((y, y_original[ind]))

    return X, y


def fit_model(loss_fun, X_, y_, learning_rate=1,
              iter_num=2000, elems=50, decr=1.001, verbosity=1):
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

        optimal_weight, loss = SGD(loss_fun, X, yy, init_w, learning_rate, iter_num, elems, decr, verbosity=verbosity)

        optimal_weights.append(optimal_weight)
        losses.append(loss)

    return np.array(optimal_weights), np.array(losses)


def learning_model(X, y, K, verbosity=1, iter_num=3000):
    C_coef = 5000
    regul_loss = regul_loss_in_point(hinge_loss_in_point, 0.5 / C_coef, ord_=1)

    if K > 1:
        K_weights = []
        K_accuracy = []
        kf = KFold(n_splits=K, shuffle=True, random_state=7)

        for i, (train_index, val_index) in enumerate(kf.split(X)):
            print(f'K = {i+1}/{K}\n')
            weights, losses = fit_model(regul_loss, X[train_index], y[train_index],
                                        learning_rate=1, iter_num=iter_num, elems=100,
                                        decr=1.001, verbosity=verbosity
                                        )
            K_weights.append(weights)
            K_accuracy.append(accuracy(X[val_index], y[val_index], weights))
            print('\n')

        save_weights(f'weights/K_weights', K_weights)
        optimal_weights = K_weights[np.argmax(K_accuracy)]
    else:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
        optimal_weights, _ = fit_model(regul_loss, X_train, y_train,
                                       learning_rate=1, iter_num=iter_num, elems=100,
                                       decr=1.001, verbosity=verbosity
                                       )
    return optimal_weights


def parse_args():
    path_to_x_train = 'samples/train-images-idx3-ubyte.gz'
    path_to_y_train = 'samples/train-labels-idx1-ubyte.gz'
    path_to_model = 'samples/my_model'

    parser = argparse.ArgumentParser()
    parser.add_argument('-x', '--x_train_dir', default=path_to_x_train,
                        help=f'путь к файлу, в котором лежат рекорды обучающей выборки, '
                             f'по умолчанию: {path_to_x_train}')
    parser.add_argument('-y', '--y_train_dir', default=path_to_y_train,
                        help=f'путь к файлу, в котором лежат метки обучающей выборки, '
                             f'по умолчанию: {path_to_y_train}')
    parser.add_argument('-m', '--model_output_dir', default=path_to_model,
                        help='путь к файлу, в который скрипт сохраняет обученную модель, '
                             f'по умолчанию: {path_to_model}')
    parser.add_argument('-v', '--verbosity', default=1,
                        help='отображение хода обучения, по умолчанию: 1')
    parser.add_argument('-k', '--k', type=int, default=1,
                        help='параметр для Кросс-валидации, по умолчанию: 1')
    parser.add_argument('-i', '--iter_num', type=int, default=3000,
                        help='количество итераций, по умолчанию: 3000')

    return parser.parse_args()


def main():

    args = parse_args()

    path_to_x_train = args.x_train_dir
    path_to_y_train = args.y_train_dir
    path_to_model = args.model_output_dir
    verbosity = args.verbosity
    K = args.k
    iter_num = args.iter_num

    X_original = read_mnist(path_to_x_train)
    y_original = read_mnist(path_to_y_train)

    X = preprocessing_data(X_original)
    y = y_original.astype(np.int8)

    optimal_weights = learning_model(X, y, K, verbosity, iter_num)
    print(f'Saving model to {path_to_model}')
    save_weights(path_to_model, optimal_weights)

    print('Metrics on the train data:\n')
    prediction_by_weights(X, y, optimal_weights, 4)


if __name__ == "__main__":
    main()
