#!/usr/bin/env python

"""
Implements variants of restricted Boltzmann machines trained with contrastive divergence (CD).
"""

import numpy as np

from copy import deepcopy
from utils import get_cosine_similarity, get_mean_squared_error


def logistic_function(x):
    """
    Numerically stable sigmoid.
    https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/

    Maps x values in the range [-inf, +inf] to y values in the range [0, 1].
    """
    positive = x >= 0.
    negative = ~positive
    y = np.zeros_like(x)
    y[positive] = 1.                  / (1. + np.exp(-x[positive]))
    y[negative] = np.exp(x[negative]) / (np.exp(x[negative]) + 1.)
    return y


class LogisticLayer(object):

    def __repr__(self):
        return "{} logistic units".format(self.count)

    def __init__(self, count):
        self.count = count

    def activation_function(self, x):
        return logistic_function(x)


class BoltzmannLayer(LogisticLayer):

    def __repr__(self):
        return "{} Boltzmann units".format(self.count)

    def activation_function(self, x):
        p = super(BoltzmannLayer, self).activation_function(x)
        r = np.random.rand(*p.shape)
        return (p > r).astype(np.int)


class RestrictedBoltzmannMachine(object):

    def __repr__(self):
        return "RBM"

    def __init__(self, layers, scale_weights_by=1.):
        self.layers  = layers
        self._initialize_weights(scale_weights_by=scale_weights_by)
        self._initialize_biases()

    def _initialize_weights(self, scale_weights_by=1.):
        for ii in range(len(self.layers)-1):
            w = self._initialize_weight_matrix(self.layers[ii].count,
                                               self.layers[ii+1].count,
                                               scale_weights_by=scale_weights_by)
            self.layers[ii].forward_weights    = w
            self.layers[ii+1].backward_weights = w.T

    def _initialize_weight_matrix(self, total_sources, total_targets, scale_weights_by=1.):
        return scale_weights_by * np.random.randn(total_sources, total_targets)

    def _initialize_biases(self):
        for ii in range(len(self.layers)):
            self.layers[ii].biases = np.zeros((self.layers[ii].count))

    def _forward_pass(self, activities, layers):
        for ii in range(len(layers)-1):
            inputs = np.dot(activities[ii], layers[ii].forward_weights)
            activities[ii+1] = layers[ii+1].activation_function(inputs + layers[ii+1].biases)
        return activities

    def _backward_pass(self, activities, layers):
        for ii in range(len(layers)-1, 0, -1):
            inputs = np.dot(activities[ii], layers[ii].backward_weights)
            activities[ii-1] = layers[ii-1].activation_function(inputs + layers[ii-1].biases)
        return activities

    def _run(self, batch, layers, cd=1):
        # initialize activities
        total_samples = len(batch)
        activities = [np.zeros((total_samples, layer.count)) for layer in layers]
        activities[0] = batch

        # run CD Gibbs sampling steps
        for ii in range(cd):
            # print "cd = {}".format(ii)
            activities = self._forward_pass(activities, layers)
            if ii == 0:
                activities_0 = deepcopy(activities)
            activities = self._backward_pass(activities, layers)

        activities_cd = deepcopy(activities)

        return activities_0, activities_cd

    def _update_weights(self, activities_0, activities_cd, layers, eta):
        for ii in range(len(layers)-1):
            vh_0  = activities_0[ii][:,:,None]  * activities_0[ii+1][:,None,:]
            vh_cd = activities_cd[ii][:,:,None] * activities_cd[ii+1][:,None,:]
            dw = eta * np.mean(vh_0 - vh_cd, axis=0)
            # print "|abs(\delta w)| : {}".format(np.mean(np.abs(dw)))
            layers[ii].forward_weights += dw
            layers[ii+1].backward_weights += dw.T

        return layers

    def _update_biases(self, activities_0, activities_cd, layers, eta):
        for ii in range(len(layers)):
            db = eta * np.mean(activities_0[ii] - activities_cd[ii], axis=0)
            layers[ii].biases += db
        return layers

    def train_layer_pair(self, inputs, visible, hidden, cd=1, eta=0., *args, **kwargs):
        """
        Train a visible, hidden layer pair.

        Arguments:
        ----------
        inputs : (batches, samples, features) ndarray
        visible, hidden : BoltzmannLayer instances
        cd : int (default 1)
        eta : float (default 0.)

        Returns:
        --------
        visible, hidden
        """

        for batch in inputs:
            activities_0, activities_cd = self._run(batch, [visible, hidden], cd)
            visible, hidden = self._update_weights(activities_0, activities_cd, [visible, hidden], eta, *args, **kwargs)
            visible, hidden = self._update_biases(activities_0, activities_cd, [visible, hidden], eta, *args, **kwargs)

        return visible, hidden

    def _apply_transform(inputs, layers):
        """
        Apply layer non-linearities to a set of inputs.
        The resulting outputs can be used to train the next (un-trained) pair of layers.
        """

        # initialize activities
        total_batches, total_samples, _ = inputs.shape
        activities = [np.zeros((total_samples, layer.count)) for layer in layers]

        # apply transform defined by the layer pair
        outputs = np.zeros((total_batches, total_samples, layers[-1].count))
        for ii, batch in enumerate(inputs):
            activities[0] = layers[0].activation_function(batch + layers[0].biases)
            activities = self._forward_pass(activities, layers)
            outputs[ii] = activities[-1].copy()

        return outputs

    def train(self, inputs, cd=1, eta=0., *args, **kwargs):
        """
        Train a stack of layers (n>2) in a bottom-up / greedy fashion:
        1) Train the lowest pair of layers using the training inputs.
        2) Create a new set of inputs by sampling from the activity in the upper layer.
        3) Repeat step 1) with the next pair of layers.

        Arguments:
        ----------
        inputs : (batches, samples, features) ndarray
        cd : int (default 1)
        eta : float (default 0.)

        Returns:
        --------
        None (updates attributes of self.layers)
        """

        for ii in range(len(self.layers)-1):
            # train a pair of layers
            visible = self.layers[ii]
            hidden = self.layers[ii+1]

            visible, hidden = self.train_layer_pair(inputs, visible, hidden, cd, eta, *args, **kwargs)

            self.layers[ii] = visible
            self.layers[ii+1] = hidden

            if ii == (len(self.layers)-2): # i.e. processing the last layer pair
                break

            # sample activities in the positive / data phase from upper layer
            # to provide training samples for the next pair of layers
            inputs = self._apply_transform(inputs, [visible, hidden])

    def _test(self, inputs, layers, loss_function, plot_function):

        forward_pass_activities, backward_pass_activities = self._run(inputs, layers, cd=1)

        a = forward_pass_activities[0] # samples x features
        b = backward_pass_activities[0] # samples x features
        loss_by_sample = loss_function(a, b, axis=-1)

        if plot_function:
            plot_function(forward_pass_activities, backward_pass_activities, layers)

        return loss_by_sample.mean()

    def test(self, inputs, loss_function=get_cosine_similarity, plot_function=None):
        return self._test(inputs, self.layers, loss_function, plot_function)

    def test_layer_pair(self, inputs, visible, hidden, loss_function=get_cosine_similarity, plot_function=None):
        return self._test(inputs, [visible, hidden], loss_function, plot_function)


class DirectedRBM(RestrictedBoltzmannMachine):
    """
    RBM variant, where the forward and backward weights are distinct and independently initialised.
    Furthermore, weight (and bias) updates can be turned on or off.
    The behaviour is controlled by passing `update_forward=True` or `update_backward=True`
    (or `update_biases`) to `train`.
    """

    def __repr__(self):
        return "Directed RBM"

    def __init__(self, layers, scale_forward_weights_by=1., scale_backward_weights_by=1.):
        self.layers  = layers
        self._initialize_weights(scale_forward_weights_by=scale_forward_weights_by,
                                 scale_backward_weights_by=scale_backward_weights_by)
        self._initialize_biases()

    def _initialize_weights(self, scale_forward_weights_by=1., scale_backward_weights_by=1.):
        for ii in range(len(self.layers)-1):
    def _update_weights(self, activities_0, activities_cd, layers, eta,
            w1 = self._initialize_weight_matrix(self.layers[ii].count,
                                                self.layers[ii+1].count,
                                                scale_weights_by=scale_forward_weights_by)
            w2 = self._initialize_weight_matrix(self.layers[ii].count,
                                                self.layers[ii+1].count,
                                                scale_weights_by=scale_backward_weights_by)
            self.layers[ii].forward_weights    = w1
            self.layers[ii+1].backward_weights = w2.T

                        update_forward=True, update_backward=True, *args, **kwargs):

        for ii in range(len(layers)-1):
            vh_0  = activities_0[ii][:,:,None]  * activities_0[ii+1][:,None,:]
            vh_cd = activities_cd[ii][:,:,None] * activities_cd[ii+1][:,None,:]
            dw = eta * np.mean(vh_0 - vh_cd, axis=0)
            # print "|abs(\delta w)| : {}".format(np.mean(np.abs(dw)))

            if update_forward:
                layers[ii].forward_weights += dw

            if update_backward:
                layers[ii+1].backward_weights += dw.T

        return layers

    def _update_biases(self, activities_0, activities_cd, layers, eta,
                       update_biases=True, *args, **kwargs):

        if update_biases:
            return super(DirectedRBM, self)._update_biases(activities_0, activities_cd, layers, eta)
