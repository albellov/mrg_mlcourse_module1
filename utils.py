import matplotlib.pyplot as plt
import numpy as np
import pickle
import struct
import gzip


def _parse_file(path, descriptor):
    with descriptor(path, 'rb') as f:
        size = struct.unpack('>xxxB', f.read(4))[0]

        shape = struct.unpack(f'>{"I"*size}', f.read(4 * size))

        shape = (shape[0], shape[1] * shape[2]) if len(shape) == 3 else (shape[0],)
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

    return data


def read_mnist(path):
    if path.endswith('.gz'):
        descriptor = gzip.open
    else:
        descriptor = open

    return _parse_file(path, descriptor)


def draw_data(X, y, rows=2, columns=8):
    for _ in range(rows):
        plt.figure(figsize=(18, 7))
        for i, n in enumerate(np.random.randint(0, len(X), columns)):
            plt.subplot(1, columns, i + 1)
            plt.title(y[n], fontsize=15)
            number = np.reshape(X[n], (28, 28))
            plt.imshow(number)

        plt.show()


def draw_mask(optimal_weights):
    plt.figure(figsize=(18, 7))
    for i, num_weight in enumerate(optimal_weights):
        plt.subplot(2, 5, i + 1)
        img1_2d = np.reshape(num_weight[1:], (28, 28))
        plt.imshow(img1_2d)
        plt.title(f'Mask for {i}', fontsize=12)
    plt.show()


def draw_loss(losses):
    plt.figure(figsize=(18, 14))
    for i, loss in enumerate(losses):
        plt.subplot(5, 2, i + 1)
        plt.title(f'Loss function with $L_{1}$ regularization for {i}', fontsize=10)
        plt.xlabel('iter_num', fontsize=15)
        plt.ylabel('loss', fontsize=15)
        plt.ylim([np.min(loss) - 0.01, np.min(loss) + 0.3])
        plt.plot(range(len(loss)), loss, 'b')
        plt.grid()
    plt.show()


def _binarization(X, threshold):
    ind = np.where(X < threshold)
    X[ind] = 0
    ind = np.where(X != 0)
    X[ind] = 1
    return X


def _scaling(X):
    return X / 255


def _add_bias(X):
    record_num, _ = X.shape
    bias_f = np.ones((record_num, 1))
    return np.hstack((bias_f, X))


def preprocessing_data(X, threshold=None):
    if threshold:
        X = _binarization(X, threshold)
    else:
        X = _scaling(X)

    return _add_bias(X)


def load_weights(file, encoding='latin1'):
    with open(file, 'rb') as f:
        return np.array(pickle.load(f, encoding=encoding))


def save_weights(filename, weights):
    with open(filename, 'wb') as f:
        pickle.dump(weights, f)
