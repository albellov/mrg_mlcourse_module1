import numpy as np
import time


class _MyKernel:
    def __init__(self, kernel_type, image_shape):
        if isinstance(kernel_type, str):
            if kernel_type == 'linear':
                self.kernel = self._linear()
                self.kernel_type = 'linear'
            elif kernel_type == 'poly':
                self.kernel = self._poly()
                self.kernel_type = 'poly'
            else:
                raise ValueError('Wrong kernel mod !')

        elif isinstance(kernel_type, function):
            self.kernel = kernel_type
        else:
            raise ValueError('Wrong kernel mod !')

        self._image_shape = image_shape

    def dot(self, X, Y, **kwargs):
        return self.kernel(X, Y, **kwargs)

    def get_features_num(self, X):
        if self.kernel_type == 'linear':
            return len(X)
        else:
            X = X.reshape(1, len(X))
            return self._add_poly_features(X).shape[1]

    def _add_poly_features(self, X):
        """Multiply selected features."""
        poly_indexes = self._index_prepare()

        i_num = 5
        k_num = 40
        alt = 3

        X_new = X**2
        X = X[:, 1:]
        for i in range(i_num):
            ci = poly_indexes[i:-6 + i:5]
            for k in range(k_num):
                ck = poly_indexes[k*alt + 1:1 + len(ci) + k*alt]
                X_new = np.hstack((X_new, X[:, ci] * X[:, ck]))

        return X_new

    def _linear(self):
        def kernel(X, Y, transpose=False):
            if transpose:
                return np.dot(X.T, Y)
            else:
                return np.dot(X, Y)
        return kernel

    def _poly(self):
        def kernel(X, Y, transpose=False):
            if transpose:
                return np.dot(self._add_poly_features(X).T, Y)
            else:
                return np.dot(self._add_poly_features(X), Y)
        return kernel

    def _index_prepare(self):
        """Select of features indexes to be multiplied."""
        height, width = self._image_shape
        middle = int(width / 2) + 2
        offsets = [5, 4]
        new_shape = height * sum(offsets)

        indexes = np.arange(height * width).reshape(self._image_shape)

        selected = range(middle - offsets[0], middle + offsets[1])
        indexes = indexes[:, selected].reshape(1, new_shape)[0]

        return indexes


class MySvm:

    def __init__(self, kernel_type='poly', C_coef=1e4, **kwargs):
        self.kernel = _MyKernel(kernel_type, **kwargs)
        self._C_coef = C_coef

        self._weights = None
        self._features_num = None

    def load_weights(self, weights):
        self._weights = weights

    def fit(self, X, y, iter_num=1000, batch_size=10, learning_rate=1, decr=1.0002, verbosity=True):
        """Train the model using SVM."""
        self._features_num = self.kernel.get_features_num(X[0, :])
        labels_num = np.unique(y).shape[0]
        weights_shape = (labels_num, self._features_num)

        self._weights = self._init_weights(weights_shape, -0.1, 0.1)

        self._mini_batch_GD(X, y, learning_rate,
                            iter_num, batch_size, decr, verbosity=verbosity)

    def _mini_batch_GD(self, X, y, learning_rate, iter_num, batch_size,
                       decr, verbosity, min_learning_rate=1e-3):
        """Mini-Batch Gradient Descent with a constant learning_rate decrease by the value of 'decr'."""

        t = time.time()
        for i in range(iter_num):
            indexes = np.random.choice(range(len(X)), batch_size)

            cur_loss, grad = self._hinge_loss_grad(X[indexes], y[indexes])
            self._weights -= learning_rate * grad

            learning_rate = max(min_learning_rate, learning_rate / decr)
            if verbosity and (i % 100 == 0 or i == iter_num - 1):
                t_end = time.time() - t
                print(f'\titer_num: {i},\tlearning_rate: {learning_rate:2.3f},'
                      f'\tloss_on_batch: {cur_loss:2.3f},\ttime for iters: {t_end:2.3} sec')
                t = time.time()

    def _hinge_loss_grad(self, X, y, **kwargs):
        """Analytical computing the gradient on the Multiclass hinge loss."""
        batch = X.shape[0]

        special_indexes = (range(batch), y)

        scores = self.kernel.dot(X, self._weights.T, **kwargs)
        correct_class_score = scores[special_indexes]
        margins = np.maximum(0, scores - correct_class_score[:, np.newaxis] + 1)
        margins[special_indexes] = 0

        # Loss and regularization
        loss = np.mean(margins) + 0.5 / self._C_coef * np.linalg.norm(self._weights, 1)**2

        margins[margins > 0] = 1
        np_sup_zero = np.sum(margins, axis=1)
        margins[special_indexes] = -np_sup_zero
        grad_w = self.kernel.dot(X, margins, transpose=True) / batch + np.sign(self._weights.T) / self._C_coef
        return loss, grad_w.T

    def predict(self, X, **kwargs):
        assert self._weights is not None, 'Need fitted model)'
        predict_labels = self.kernel.dot(X, self._weights.T, **kwargs).T
        return np.argmax(predict_labels, axis=0)

    def get_weights(self):
        return self._weights

    @staticmethod
    def _init_weights(shape, a, b):
        return np.random.uniform(a, b, shape)
