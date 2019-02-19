#!/usr/bin/env python

"""
Implements variants of restricted Boltzmann machines trained with contrastive divergence (CD).
"""

import numpy as np

from sys import stdout
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
        # version 1)
        activities = self._forward_pass(activities, layers)
        activities_0 = deepcopy(activities)
        for ii in range(cd):
            activities = self._backward_pass(activities, layers)
            activities = self._forward_pass(activities, layers)
        activities_cd = deepcopy(activities)

        # # run CD Gibbs sampling steps
        # # version 2)
        # for ii in range(cd):
        #     activities = self._forward_pass(activities, layers)
        #     if ii == 0:
        #         activities_0 = deepcopy(activities)
        #     activities = self._backward_pass(activities, layers)
        # activities_cd = deepcopy(activities)

        return activities_0, activities_cd

    def _update_weights(self, layers, activities_0, activities_cd, eta):
        for ii in range(len(layers)-1):
            vh_0  = activities_0[ii][:,:,None]  * activities_0[ii+1][:,None,:]
            vh_cd = activities_cd[ii][:,:,None] * activities_cd[ii+1][:,None,:]
            dw = eta * np.mean(vh_0 - vh_cd, axis=0)
            # print "|abs(\delta w)| : {}".format(np.mean(np.abs(dw)))
            layers[ii].forward_weights += dw
            layers[ii+1].backward_weights += dw.T

        return layers

    def _update_biases(self, layers, activities_0, activities_cd, eta):
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
            visible, hidden = self._update_weights([visible, hidden], activities_0, activities_cd, eta, *args, **kwargs)
            visible, hidden = self._update_biases([visible, hidden], activities_0, activities_cd, eta, *args, **kwargs)

        return visible, hidden

    def _apply_transform(self, inputs, layers):
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

    # def train(self, inputs, cd=1, eta=0., *args, **kwargs):
    #     """
    #     Train a stack of layers (n>2) in a bottom-up / greedy fashion:
    #     1) Train the lowest pair of layers using the training inputs.
    #     2) Create a new set of inputs by sampling from the activity in the upper layer.
    #     3) Repeat step 1) with the next pair of layers.

    #     Arguments:
    #     ----------
    #     inputs : (batches, samples, features) ndarray
    #     cd : int (default 1)
    #     eta : float (default 0.)

    #     Returns:
    #     --------
    #     None (updates attributes of self.layers)
    #     """

    #     for ii in range(len(self.layers)-1):
    #         # train a pair of layers
    #         visible = self.layers[ii]
    #         hidden = self.layers[ii+1]

    #         visible, hidden = self.train_layer_pair(inputs, visible, hidden, cd, eta, *args, **kwargs)

    #         self.layers[ii] = visible
    #         self.layers[ii+1] = hidden

    #         if ii == (len(self.layers)-2): # i.e. processing the last layer pair
    #             break

    #         # sample activities in the positive / data phase from upper layer
    #         # to provide training samples for the next pair of layers
    #         inputs = self._apply_transform(inputs, [visible, hidden])

    def train(self, inputs_train, inputs_test, test_at, test_params, cd=1, eta=0., *args, **kwargs):
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
        loss :
        """

        loss = np.full((len(self.layers)-1, len(test_at)), np.nan)

        # loop over successive layer pairs and train one pair at a time
        for ii in range(len(self.layers)-1):
            visible = self.layers[ii]
            hidden = self.layers[ii+1]

            total_batches_trained = 0
            for jj, next_total_batches_trained in enumerate(test_at):

                if next_total_batches_trained - total_batches_trained > 0: # i.e. skip edge case where we are not training at all

                    indices = np.arange(total_batches_trained, next_total_batches_trained)
                    super_batch = np.take(inputs_train, indices=indices, axis=0, mode='wrap')

                    visible, hidden = self.train_layer_pair(super_batch, visible, hidden, cd, eta, *args, **kwargs)
                    self.layers[ii] = visible
                    self.layers[ii+1] = hidden

                    total_batches_trained = next_total_batches_trained

                loss[ii, jj] = self._test(inputs_test, layers=self.layers[:ii+2],  **test_params) # i.e. including the ith + 1 layer

                # stdout.write('\r{:5d} of {:5d} total batches;'.format(rep*total_batches+test_at[ii], total_batches*total_repetitions))
                stdout.write('\r    layers: {:1d} + {:1d}; batches trained: {:6d};'.format(ii, ii+1, total_batches_trained))
                stdout.write(' loss: {:.3f}'.format(loss[ii, jj]))
                stdout.flush()

            # new line after finishing training for the current layer pair
            stdout.write('\n')
            stdout.flush()

            if ii < (len(self.layers)-2): # i.e. we have not processed the last layer pair yet
                # sample activities in the positive / data phase from previously trained layers
                # to provide training samples for the next pair of layers
                inputs_train = self._apply_transform(inputs_train, [visible, hidden])

        return loss.ravel()

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
            w1 = self._initialize_weight_matrix(self.layers[ii].count,
                                                self.layers[ii+1].count,
                                                scale_weights_by=scale_forward_weights_by)
            w2 = self._initialize_weight_matrix(self.layers[ii].count,
                                                self.layers[ii+1].count,
                                                scale_weights_by=scale_backward_weights_by)
            self.layers[ii].forward_weights    = w1
            self.layers[ii+1].backward_weights = w2.T

    def _update_weights(self, layers, activities_0, activities_cd, eta,
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

    def _update_biases(self, layers, activities_0, activities_cd, eta,
                       update_biases=True, *args, **kwargs):
        if update_biases:
            return super(DirectedRBM, self)._update_biases(layers, activities_0, activities_cd, eta)


class SparselyConnected(object):

    def _initialize_connectivity_matrix(self, total_sources, total_targets, connection_probability=1.):
        return np.random.rand(total_sources, total_targets) <= connection_probability

    def _update_weights(self, layers, *args, **kwargs):
        layers = super(SparselyConnected, self)._update_weights(layers, *args, **kwargs)
        self._enforce_sparsity(layers)
        return layers

    def _enforce_sparsity(self, layers):
        for ii in range(len(layers)-1):
            layers[ii].forward_weights *= layers[ii].forward_connectivity
            layers[ii+1].backward_weights *= layers[ii+1].backward_connectivity

        return layers


class SparseRBM(SparselyConnected, RestrictedBoltzmannMachine):

    def __repr__(self):
        return "Sparsely connected RBM"

    def __init__(self, connection_probability=1., *args, **kwargs):
        super(SparseRBM, self).__init__(*args, **kwargs)
        self._initialize_connectivity(connection_probability=connection_probability)

    def _initialize_connectivity(self, connection_probability=1.):
        for ii in range(len(self.layers)-1):
            c = self._initialize_connectivity_matrix(self.layers[ii].count,
                                                     self.layers[ii+1].count,
                                                     connection_probability=connection_probability)
            self.layers[ii].forward_connectivity = c
            self.layers[ii+1].backward_connectivity = c.T


class SparseDirectedRBM(SparselyConnected, DirectedRBM):

    def __repr__(self):
        return "Sparsely connected directed RBM"

    def __init__(self, connection_probability=1., *args, **kwargs):
        super(SparseDirectedRBM, self).__init__(*args, **kwargs)
        self._initialize_connectivity(connection_probability=connection_probability)

    def _initialize_connectivity(self, connection_probability=1.):
        for ii in range(len(self.layers)-1):
            c1 = self._initialize_connectivity_matrix(self.layers[ii].count,
                                                     self.layers[ii+1].count,
                                                     connection_probability=connection_probability)
            c2 = self._initialize_connectivity_matrix(self.layers[ii].count,
                                                     self.layers[ii+1].count,
                                                     connection_probability=connection_probability)
            self.layers[ii].forward_connectivity = c1
            self.layers[ii+1].backward_connectivity = c2.T


class ComplementaryRBM(SparseDirectedRBM):

    def __repr__(self):
        return "Sparsely connected directed RBM in which no connection is bi-directional"

    def _initialize_connectivity(self, connection_probability=0.5):

        assert connection_probability <= 0.5, "Cannot construct complementary connectivity with p>0.5!"

        for ii in range(len(self.layers)-1):
            c1 = self._initialize_connectivity_matrix(self.layers[ii].count,
                                                     self.layers[ii+1].count,
                                                     connection_probability=connection_probability)
            c2 = self._sample_complementary_connectivity_matrix(c1, connection_probability)
            self.layers[ii].forward_connectivity = c1
            self.layers[ii+1].backward_connectivity = c2.T

    def _sample_complementary_connectivity_matrix(self, connectivity_matrix, desired_connection_probability):
        complement = np.invert(connectivity_matrix)
        actual_connection_probability = np.mean(complement)
        corrected_probability = desired_connection_probability / actual_connection_probability
        resample_by = np.random.rand(*complement.shape) <= corrected_probability
        return complement * resample_by


class SparselyActiveDirectedRBM(DirectedRBM):

    def _get_penalty(self, activities_0, activities_cd,
                     sparsity_target=0.1, sparsity_cost=1.):

        # only ever want to impose sparsity bias on hidden layer,
        # which is presumed to be the last; hence the -1 index;
        # TODO: compute q as a temporally averaged version via q = lambda * q_old + (1-lambda) * q_current
        q = (np.mean(activities_0[-1], axis=0) + np.mean(activities_cd[-1], axis=0))/2
        penalty = sparsity_cost * (sparsity_target - q)
        return penalty

    def _update_weights(self, layers, activities_0, activities_cd, eta,
                        update_forward=True, update_backward=True,
                        sparsity_target=0.1, sparsity_cost=1.,
                        *args, **kwargs):

        layers = super(SparselyActiveDirectedRBM, self)._update_weights(layers, activities_0, activities_cd, eta,
                                                                        update_forward, update_backward,
                                                                        *args, **kwargs)

        penalty = self._get_penalty(activities_0, activities_cd, sparsity_target, sparsity_cost)

        if update_forward:
            layers[0].forward_weights   += eta * penalty[None,:] * np.ones_like(layers[0].forward_weights)

        if update_backward:
            layers[-1].backward_weights += eta * penalty[:,None] * np.ones_like(layers[-1].backward_weights)

        return layers

    def _update_biases(self, layers, activities_0, activities_cd, eta,
                       update_biases=True,
                       sparsity_target=0.1, sparsity_cost=1.,
                       *args, **kwargs):

        layers = super(SparselyActiveDirectedRBM, self)._update_biases(layers, activities_0, activities_cd, eta,
                                                                       update_biases=update_biases,
                                                                       *args, **kwargs)

        if update_biases:
            penalty = self._get_penalty(activities_0, activities_cd, sparsity_target, sparsity_cost)
            layers[-1].biases += eta * penalty

        return layers
