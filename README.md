# mrg_mlcourse_module1
## Solving steps
1. Using Linear SVM-model with One-vs-All strategy. (`F1-score = 0.9079`)
2. Added Cross-Validation. (`F1-score = 0.9137`, it was a bad idea)
3. Using SVM-model with Multiclass hinge loss. (http://cs231n.github.io/linear-classify/)
4. Added pre-processing features:
    - Сrop 3 pixels at top and bottom, 6 pixels at the sides. The features num became equal to 252.
    - Binarization with threshold. It got worse, now not used.
5. Added analytical computing the gradient on the Multiclass hinge loss. So more faster! (https://twice22.github.io/hingeloss/)
6. Added polynomial features. (https://en.wikipedia.org/wiki/Polynomial_kernel)
    - Increased features num to 7000.
    
## Metrics
* Valid set(last 10000 train's set records):
    ``
* Test set:
    ``
## Using
**Training**
```shell
$ usage: python train.py [-h] [-x X_TRAIN_DIR] [-y Y_TRAIN_DIR] [-m MODEL_OUTPUT_DIR]
                [-v VERBOSITY] [-i ITER_NUM] [-b BATCH_SIZE] [-k KERNEL]

optional arguments:
  -h, --help            show this help message and exit
  -x X_TRAIN_DIR, --x_train_dir X_TRAIN_DIR
                        path to the file with the training sample's records,
                        default: samples/train-images-idx3-ubyte.gz
  -y Y_TRAIN_DIR, --y_train_dir Y_TRAIN_DIR
                        path to the file with the training sample's labels,
                        default: samples/train-labels-idx1-ubyte.gz
  -m MODEL_OUTPUT_DIR, --model_output_dir MODEL_OUTPUT_DIR
                        path to the file for saving model, default:
                        samples/my_model_multi
  -v VERBOSITY, --verbosity VERBOSITY
                        is verbosity, default: 1
  -i ITER_NUM, --iter_num ITER_NUM
                        number of iterations, default: 3000
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        mini-batch size, default: 20
  -k KERNEL, --kernel KERNEL
                        kernel function: 'linear' or 'poly', default: 'poly'
```

**Predict**
```shell
$ usage: python predict.py [-h] [-x X_TEST_DIR] [-y Y_TEST_DIR] [-m MODEL_INPUT_DIR]
                  [-k KERNEL]

optional arguments:
        -h, --help            show this help message and exit
        -x X_TEST_DIR, --x_test_dir X_TEST_DIR
                        path to the file with the testing sample's records,
                        default: samples/t10k-images-idx3-ubyte.gz
        -y Y_TEST_DIR, --y_test_dir Y_TEST_DIR
                        path to the file with the testing sample's labels,
                        default: samples/t10k-labels-idx1-ubyte.gz
        -m MODEL_INPUT_DIR, --model_input_dir MODEL_INPUT_DIR
                        path to the file for loading model, default:
                        samples/my_model
        -k KERNEL, --kernel KERNEL
                        kernel function: 'linear' or 'poly', default: 'poly'
```