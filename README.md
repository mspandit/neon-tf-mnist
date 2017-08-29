# MNIST in Neon and Tensorflow

This repository includes implementations of a deep learning model using two
different frameworks: Intel Nervana
[Neon](http://neon.nervanasys.com/docs/latest/index.html) and Google
[Tensorflow](https://www.tensorflow.org/). 

These implementations are intended to illustrate the differences in the
programming models presented by the two frameworks.

# The Problem

The model solves an old problem from the machine learning community: assign a
28 &times; 28 pixel grayscale image of a handwritten digit to the correct one
of ten classes.

The model is trained and tested on 70,000 images from the [MNIST
database](https://en.wikipedia.org/wiki/MNIST_database).

# The Implementations

## Parsing Arguments

The Neon implementation uses a [`NeonArgparser`](http://neon.nervanasys.com/docs/latest/generated/neon.util.argparser.NeonArgparser.html) instance to parse command-line arguments:

```
if __name__ == '__main__':
    # parse the command line arguments
    main(NeonArgparser(__doc__).parse_args())
```

The Tensorflow implementation uses an `[ArgumentParser](https://docs.python.org/2/library/argparse.html#argumentparser-objects)` instance. The
`data_dir` argument specifies the location of cached training data (if any).

It then calls our `main()` function, providing command-line arguments.

```

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/mnist_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
```

## Preparing Data

The Neon implementation uses an [MNIST](http://neon.nervanasys.com/docs/latest/datasets.html#mnist) instance to aquire data sets. The
[MNIST](http://neon.nervanasys.com/docs/latest/datasets.html#mnist) instance handles downloading the MNIST database into a local cache.

```
    dataset = MNIST(path=args.data_dir)
```

The Tensorflow implementation uses an [mnist.input_data](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/tutorials/mnist/?hl=fr) instance. 

```
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
```

## Defining the Model

The Neon implementation defines the model as two [Affine](http://neon.nervanasys.com/docs/latest/layers.html#compound-layers) layers with
[Gaussian](http://neon.nervanasys.com/docs/latest/initializers.html) initialization. The first layer has a [rectified linear](http://neon.nervanasys.com/docs/latest/activations.html) activation,
and the second a [Logistic](http://neon.nervanasys.com/docs/latest/activations.html) activation.

The model is instantiated directly with these two layers.

```
    mlp = Model(
        layers=[
            Affine(nout=100, init=Gaussian(loc=0.0, scale=0.01),
                   activation=Rectlin()),
            Affine(nout=10, init=Gaussian(loc=0.0, scale=0.01),
                   activation=Logistic(shortcut=True))])
```

The Tensorflow implementation defines the model as a collection of

* [placeholder](https://www.tensorflow.org/api_docs/python/tf/placeholder)s, 
* [Variables](https://www.tensorflow.org/api_docs/python/tf/Variable) initialized using [random_normal_initializer](https://www.tensorflow.org/api_docs/python/tf/random_normal_initializer), 
* [matrix multiplication](https://www.tensorflow.org/api_docs/python/tf/matmul) operations, and
* [rectified linear](https://www.tensorflow.org/api_docs/python/tf/nn/relu) activations. 

These objects are actually references into a graph representation of the model.
This representation expresses the dependencies between the outputs, various
intermediate values, inputs, and the matrix operations on them.

```
    x = tf.placeholder(tf.float32, [None, 784])
    W1 = tf.Variable(tf.random_normal_initializer()([784, 100]))
    b1 = tf.Variable(tf.random_normal_initializer()([100]))
    W2 = tf.Variable(tf.random_normal_initializer()([100, 10]))
    b2 = tf.Variable(tf.random_normal_initializer()([10]))
    y = tf.matmul(tf.nn.relu(tf.matmul(x, W1) + b1), W2) + b2
```

## Defining the Optimizer

The Neon implementation defines the optimizer as [GradientDescentMomentum](http://neon.nervanasys.com/docs/latest/optimizers.html#stochastic-gradient-descent)

```
    optimizer = GradientDescentMomentum(
        0.1, momentum_coef=0.9, stochastic_round=args.rounding)
```

The Tensorflow implementation defines the optimizer as a collection of 

* [placeholder](https://www.tensorflow.org/api_docs/python/tf/placeholder)s for the actual and expected outputs, and
* operations including [reduce_mean](https://www.tensorflow.org/api_docs/python/tf/reduce_mean) and [softmax_cross_entropy_with_logits](https://www.tensorflow.org/versions/master/api_docs/python/tf/nn/softmax_cross_entropy_with_logits) (the cost function)

Again, these objects are references into a graph representation of the
optimizer.

```
    y_ = tf.placeholder(tf.float32, [None, 10])
    train_step = tf.train.MomentumOptimizer(0.1, 0.9).minimize(
      tf.reduce_mean(
          tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)))
```

## Fitting the Model

The Neon implementation fits the model to the training set by passing the
optimizer to its `fit()` method. The cost function is specified here using a
[GeneralizedCost](http://neon.nervanasys.com/docs/latest/generated/neon.layers.l
ayer.GeneralizedCost.html) layer and
[CrossEntropyBinary](http://neon.nervanasys.com/docs/latest/generated/neon.trans
forms.cost.CrossEntropyBinary.html) function. The number of training epochs is
derived from the command line arguments.

```
    mlp.fit(
        dataset.train_iter,
        optimizer=optimizer,
        num_epochs=args.epochs,
        cost=GeneralizedCost(costfunc=CrossEntropyBinary()),
        callbacks=callbacks)
```

The Tensorflow implementation fits the model to the training set by 

1. registering a default
[session](https://www.tensorflow.org/programmers_guide/graphs#executing_a_graph_in_a_tfsession) in the context of which to execute the graph. 

2. initializing global variables

3. acquiring a batch of training data and

4. running the optimizer with the batch mapped to placeholders in the model.

5. repeating steps 3. and 4.

```
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    for _ in range(4690):
        batch_xs, batch_ys = mnist.train.next_batch(128)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
```

## Displaying Accuracy

The Neon implementation evaluates the accuracy of the model on the validation set by calling its `eval()` method and passing the [Misclassification](http://neon.nervanasys.com/docs/latest/generated/neon.transforms.cost.Misclassification.html) metric.

```
    error_rate = mlp.eval(dataset.valid_iter, metric=Misclassification())
    neon_logger.display('Classification accuracy = %.4f' % (1 - error_rate))
```

The Tensorflow implementation defines an accuracy measurement as a collection of 

* [placeholder]()s for the actual and expected outputs, and 
* operations including [equal](https://www.tensorflow.org/versions/r1.3/api_docs/python/tf/equal) and [reduce_mean](https://www.tensorflow.org/api_docs/python/tf/reduce_mean)

These objects are actually references into a graph representation of the
accuracy formula. It evaluates the accuracy by running this graph with the test
set mapped to placeholders.

```
    accuracy = tf.reduce_mean(
        tf.cast(
            tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)), 
            tf.float32))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                        y_: mnist.test.labels}))
```

# Running

To run the Neon implementation, [follow instructions for installing Neon](http://neon.nervanasys.com/docs/latest/installation.html). Then, simply enter

```
(.venv2) :neon-tf-mnist $ python neon_mnist_mlp.py 
```

To run the Tensorflow implementation, [follow instructions for installing Tensorflow](https://www.tensorflow.org/install/). Then simply enter

```
(tensorflow) :neon-tf-mnist $ python tf_mnist_mlp.py
```