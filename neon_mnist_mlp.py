#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright 2015-2016 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
"""
Train a small multi-layer perceptron with fully connected layers on MNIST data.

This example has some command line arguments that enable different neon
features.

Examples:

    python examples/mnist_mlp.py -b gpu -e 10

        Run the example for 10 epochs using the NervanaGPU backend

    python examples/mnist_mlp.py --eval_freq 1

        After each training epoch, process the validation/test data
        set through the model and display the cost.

    python examples/mnist_mlp.py --serialize 1 -s checkpoint.pkl

        After every iteration of training, dump the model to a pickle
        file named "checkpoint.pkl".  Changing the serialize parameter
        changes the frequency at which the model is saved.

    python examples/mnist_mlp.py --model_file checkpoint.pkl

        Before starting to train the model, set the model state to
        the values stored in the checkpoint file named checkpoint.pkl.

"""

from neon.callbacks.callbacks import Callbacks
from neon.data import MNIST
from neon.initializers import Gaussian
from neon.layers import GeneralizedCost, Affine
from neon.models import Model
from neon.optimizers import GradientDescentMomentum
from neon.transforms import (Rectlin, Logistic, CrossEntropyBinary,
                             Misclassification)
from neon.util.argparser import NeonArgparser
from neon import logger as neon_logger


def main(args):
    # load up the mnist data set
    dataset = MNIST(path=args.data_dir)

    # initialize model object
    mlp = Model(
        layers=[
            Affine(nout=100, init=Gaussian(loc=0.0, scale=0.01),
                   activation=Rectlin()),
            Affine(nout=10, init=Gaussian(loc=0.0, scale=0.01),
                   activation=Logistic(shortcut=True))])

    # setup optimizer
    optimizer = GradientDescentMomentum(
        0.1, momentum_coef=0.9, stochastic_round=args.rounding)

    # configure callbacks
    callbacks = Callbacks(mlp, eval_set=dataset.valid_iter, **args.callback_args)

    # run fit
    # setup cost function as CrossEntropy
    mlp.fit(
        dataset.train_iter,
        optimizer=optimizer,
        num_epochs=args.epochs,
        cost=GeneralizedCost(costfunc=CrossEntropyBinary()),
        callbacks=callbacks)
    error_rate = mlp.eval(dataset.valid_iter, metric=Misclassification())
    neon_logger.display('Classification accuracy = %.4f' % (1 - error_rate))


if __name__ == '__main__':
    main(NeonArgparser(__doc__).parse_args())
