# mrg_mlcourse_module1
## Solving steps
1. Using Linear SVM-model with One-vs-All strategy. (`F1-score = 0.9079`)
2. Added Cross-Validation. (`F1-score = 0.9137`, it was a bad idea)
3. Using SVM-model with Multiclass hinge loss. (http://cs231n.github.io/linear-classify/)
4. Added pre-processing features:
    - Сrop 5 pixels at top and bottom, 7 pixels at the sides. The features num became equal to 252.
    - Binarization with threshold. It got worse, now not used.
5. Added analytical computing the gradient on the Multiclass hinge loss. So more faster! (https://twice22.github.io/hingeloss/)
6. Added polynomial features. (https://en.wikipedia.org/wiki/Polynomial_kernel)
    - Increased features num to 6653.
    
## Metrics
* Valid set(last 10000 train's set records):
    
              precision   recall  f1-score   support
    
          0     0.9788    0.9768    0.9778       991
          1     0.9796    0.9934    0.9865      1064
          2     0.9409    0.9646    0.9526       990
          3     0.9479    0.9544    0.9511      1030
          4     0.9774    0.9695    0.9734       983
          5     0.9677    0.9180    0.9422       915
          6     0.9427    0.9866    0.9641       967
          7     0.9754    0.9459    0.9604      1090
          8     0.9623    0.9356    0.9487      1009
          9     0.9301    0.9553    0.9425       961
        avg     0.9606    0.9603    0.9603     10000   

    
    
* Test set:
          
              precision    recall  f1-score   support

          0     0.9755    0.9755    0.9755       980
          1     0.9749    0.9938    0.9843      1135
          2     0.9333    0.9622    0.9475      1032
          3     0.9458    0.9683    0.9569      1010
          4     0.9653    0.9644    0.9648       982
          5     0.9669    0.9170    0.9413       892
          6     0.9453    0.9749    0.9599       958
          7     0.9794    0.9241    0.9510      1028
          8     0.9627    0.9538    0.9582       974
          9     0.9517    0.9574    0.9545      1009

        avg     0.9602    0.9599    0.9598     10000
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