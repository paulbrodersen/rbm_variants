#!/usr/bin/env python

"""
Train RBM variants to reconstruct MNIST digits, and
reproduce graphs in Brodersen et al (2019).
"""

import numpy as np
import matplotlib.pyplot as plt
import mnist

from functools import partial

from rbm_variants import (LogisticLayer, BoltzmannLayer,
                          RestrictedBoltzmannMachine, DirectedRBM,
                          SparseRBM, SparseDirectedRBM, ComplementaryRBM,
                          SparselyActiveDirectedRBM,
)

from utils import (rescale, make_batches, make_balanced_batches, characterise_model,
                   subplots, get_unblockedshaped,
                   get_cosine_similarity, get_mean_squared_error,
                   nanhist,
)

# styling
import matplotlib as mpl
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False

DEFAULT_COLOR = '#1f77b4'


def load_mnist(ddir='../data/mldata', mode='train'):

    # load mnist
    mndata = mnist.MNIST(ddir)

    if mode is 'train':
        inputs, targets = mndata.load_training()
    elif mode is 'test':
        inputs, targets = mndata.load_testing()
    else:
        raise ValueError("`mode` is either 'train' or 'test'")

    inputs = np.array(inputs, dtype=np.float)
    targets = np.array(targets, dtype=np.int)

    # # preprocess outputs
    # targets = one_hot_encoding(targets, 10)

    return inputs, targets


class DiagnosticPlotter(object):
    """
    Create diagnostic plots that describe
    - the current state of the RBM, and
    - changes therein over time.
    """

    def __init__(self, samples_trained, layers, figure_directory='../figures/', *args, **kwargs):

        # house keeping
        self.layers = layers

        # if the number of layers exceeds 2,
        # the full training is run one additional time for each additional layers;
        if len(layers) == 2:
            self.samples_trained = samples_trained
        else:
            self.samples_trained = np.concatenate([samples_trained + np.max(samples_trained) * ii for ii in range(len(layers)-1)])
        self.figure_directory = figure_directory
        self.args = args
        self.kwargs = kwargs

        # initialise data structures
        self._initialize_history(total_tests=len(self.samples_trained))

    def _initialize_history(self, total_tests=100):
        """
        Keep track of
        - weights
        - biases
        - forward and backward pass activities
        """
        self._initialize_weights_history(total_tests)
        self._initialize_biases_history(total_tests)
        self._initialize_activity_history(total_tests)
        # self._initialize_loss_history(total_tests)
        self.ptr = 0  # `_update_history` increments ctr by one but is called before plotting routines

    def _initialize_weights_history(self, total_tests):
        for ii in range(len(self.layers)-1):
            self.layers[ii].forward_weights_history    = np.full((total_tests, self.layers[ii].count, self.layers[ii+1].count), np.nan)
            self.layers[ii+1].backward_weights_history = np.full((total_tests, self.layers[ii+1].count, self.layers[ii].count), np.nan)

    def _initialize_biases_history(self, total_tests):
        for ii in range(len(self.layers)):
            self.layers[ii].biases_history = np.full((total_tests, self.layers[ii].count), np.nan)

    def _initialize_activity_history(self, total_tests):
        for ii in range(len(self.layers)):
            self.layers[ii].forward_pass_activity_history = np.full((total_tests, self.layers[ii].count), np.nan)
            self.layers[ii].backward_pass_activity_history = np.full((total_tests, self.layers[ii].count), np.nan)

    def _initialize_loss(self, total_tests):
        self.loss = np.full((len(self.layers)-1, len(test_at)), np.nan)

    def __call__(self, forward_pass_activities, backward_pass_activities, layers):

        self._update_history(layers,
                             forward_pass_activities,
                             backward_pass_activities)
        self.plot_state()
        self.plot_history()
        self.plot_reconstructions(forward_pass_activities[0],
                                  backward_pass_activities[0])

        self.ptr += 1

    def _update_history(self, layers, forward_pass_activities, backward_pass_activities):
        self._update_weights_history(self.ptr, layers)
        self._update_biases_history(self.ptr, layers)
        self._update_activity_history(self.ptr, layers,
                                      forward_pass_activities,
                                      backward_pass_activities)


    def _update_weights_history(self, ptr, layers):
        for ii in range(len(layers)-1):
            self.layers[ii].forward_weights_history[ptr]    = layers[ii].forward_weights
            self.layers[ii+1].backward_weights_history[ptr] = layers[ii+1].backward_weights

    def _update_biases_history(self, ptr, layers):
        for ii in range(len(layers)):
            self.layers[ii].biases_history[ptr] = layers[ii].biases

    def _update_activity_history(self, ptr, layers, forward_pass_activities, backward_pass_activities):
        for ii in range(len(layers)):
            self.layers[ii].forward_pass_activity_history[ptr] = np.mean(forward_pass_activities[ii], axis=0)
            self.layers[ii].backward_pass_activity_history[ptr] = np.mean(backward_pass_activities[ii], axis=0)

    def plot_state(self):
        self.plot_weights()
        self.plot_biases()
        self.plot_weight_alignment()
        self.plot_activity()

    def plot_weights(self,
                     bins=np.linspace(-10, 10, 21),
                     color=DEFAULT_COLOR,
                     xlim=(-10, 10),
                     ylim=(0, 120 * 1e+3),
                     figure_name='weight_distribution'):
        """
        Plot current weight distribution by layer.
        """

        fig, axes = subplots(len(self.layers)-1, 2, sharex=True, sharey=True)

        for ii, (ax1, ax2) in enumerate(axes):
            nanhist(self.layers[ii].forward_weights_history[self.ptr].ravel(), bins=bins, color=color, ax=ax1)
            nanhist(self.layers[ii+1].backward_weights_history[self.ptr].ravel(), bins=bins, color=color, ax=ax2)

        # label axes
        axes[ 0, 0].set_title("Forward weights")
        axes[ 0, 1].set_title("Backward weights")
        axes[-1, 0].set_xlabel('Weight value [AU]')
        axes[-1, 1].set_xlabel('Weight value [AU]')
        for ii, ax in enumerate(axes[:, 0]):
            ax.set_ylabel("Number of weights\nfrom layer {} to layer {}".format(ii, ii+1))

        # set axis limits
        for ax in axes.ravel():
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
        fig.tight_layout()

        fig.savefig(self.figure_directory + figure_name + ".pdf")
        fig.savefig(self.figure_directory + figure_name + ".svg")

    def plot_biases(self,
                    bins=np.linspace(-10, 10, 21),
                    color=DEFAULT_COLOR,
                    xlim=(-10, 10),
                    ylim=(0, 400),
                    figure_name='bias_distribution'):
        """
        Plot current distribution of biases.
        """

        fig, axes = plt.subplots(len(self.layers), sharex=True, sharey=True)
        for ii, ax in enumerate(axes):
            nanhist(self.layers[ii].biases_history[self.ptr], bins=bins, color=color, ax=ax)

        # label axes
        for ii, ax in enumerate(axes):
            ax.set_ylabel('Layer {}\nNumber of biases'.format(ii))
        axes[-1].set_xlabel('Bias value [AU]')

        # set axis limits
        for ax in axes.ravel():
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
        fig.tight_layout()

        fig.savefig(self.figure_directory + figure_name + ".pdf")
        fig.savefig(self.figure_directory + figure_name + ".svg")

    def plot_activity(self,
                      bins=np.linspace(0., 1., 11),
                      color=DEFAULT_COLOR,
                      xlim=(0, 1),
                      ylim=(0, 500),
                      figure_name="activity_distribution"):
        """
        Plot current activity distribution by layer.
        """

        fig, axes = plt.subplots(len(self.layers), 2, sharex=True, sharey=True)
        for ii, layer in enumerate(self.layers):
            activities = (layer.forward_pass_activity_history[self.ptr],
                          layer.backward_pass_activity_history[self.ptr])
            for jj, activity in enumerate(activities):
                nanhist(activity, bins=bins, color=color, ax=axes[ii, jj])
                # annotate mean
                p = np.mean(activity)
                axes[ii, jj].axvline(p, 0, 1, color='k', ls='--', alpha=0.1)
                axes[ii, jj].annotate(r'$p = {:.3f}$'.format(p), (p, 0.8),
                                      xycoords='axes fraction',
                                      horizontalalignment='center')

        # label axes
        axes[ 0, 0].set_title("Data phase")
        axes[ 0, 1].set_title("Model phase")
        axes[-1, 0].set_xlabel('Fraction of samples ON')
        axes[-1, 1].set_xlabel('Fraction of samples ON')
        for ii, ax in enumerate(axes[:, 0]):
            ax.set_ylabel('Layer {}\nNumber of units'.format(ii))

        # set axis limits
        for ax in axes.ravel():
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
        fig.tight_layout()

        fig.savefig(self.figure_directory + figure_name + ".pdf")
        fig.savefig(self.figure_directory + figure_name + ".svg")

    def plot_weight_alignment(self,
                      color=DEFAULT_COLOR,
                      xlim=(-10, 10),
                      ylim=(-10, 10),
                      figure_name="weight_alignment"):

        fig, axes = subplots(len(self.layers)-1, 1)
        for ii, ax in enumerate(axes.ravel()):
            ax.plot(self.layers[ii].forward_weights_history[self.ptr].ravel(),
                    self.layers[ii+1].backward_weights_history[self.ptr].transpose().ravel(),
                    '.', markersize=1, color=color, alpha=0.1,
                    rasterized=True)

        for ax in axes.ravel():
            ax.set_xlabel('Forward weights [AU]')
            ax.set_ylabel('Backward weights [AU]')

        # set axis limits
        for ax in axes.ravel():
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
        fig.tight_layout()

        fig.savefig(self.figure_directory + figure_name + ".pdf")
        fig.savefig(self.figure_directory + figure_name + ".svg")

    def plot_history(self):
        self.plot_weights_history()
        self.plot_biases_history()
        self.plot_activity_history()
        # self.plot_weight_alignment_history()

    def plot_weights_history(self, color=DEFAULT_COLOR, figure_name='weight_history'):

        fig, axes = subplots(len(self.layers)-1, 2, sharex=True, sharey=True)
        for ii, ax in enumerate(axes):
            plot_distribution_over_time(self.samples_trained[:self.ptr+1],
                                        self.layers[ii].forward_weights_history[:self.ptr+1].reshape(self.ptr+1, -1),
                                        color=color,
                                        ax=axes[ii, 0])
            plot_distribution_over_time(self.samples_trained[:self.ptr+1],
                                        self.layers[ii+1].backward_weights_history[:self.ptr+1].reshape(self.ptr+1, -1),
                                        color=color,
                                        ax=axes[ii, 1])

        # label axes
        axes[ 0, 0].set_title("Forward weights")
        axes[ 0, 1].set_title("Backward weights")
        axes[-1, 0].set_xlabel('Samples')
        axes[-1, 1].set_xlabel('Samples')
        for jj, ax in enumerate(axes[:, 0]):
            ax.set_ylabel('Layer {}\nWeight [AU]'.format(jj))

        fig.tight_layout()

        fig.savefig(self.figure_directory + figure_name + ".pdf")
        fig.savefig(self.figure_directory + figure_name + ".svg")

    def plot_biases_history(self,
                            color=DEFAULT_COLOR,
                            figure_name='biases_history'):

        fig, axes = plt.subplots(len(self.layers), 1, sharex=True, sharey=True)

        for ii, layer in enumerate(self.layers):
            plot_distribution_over_time(self.samples_trained[:self.ptr+1],
                                        layer.biases_history[:self.ptr+1],
                                        color=color,
                                        ax=axes[ii])

        # label axes
        for ii, ax in enumerate(axes):
            ax.set_ylabel('Layer {}\nBias [AU]'.format(ii))
        axes[-1].set_xlabel('Samples')

        fig.tight_layout()

        fig.savefig(self.figure_directory + figure_name + ".pdf")
        fig.savefig(self.figure_directory + figure_name + ".svg")

    def plot_activity_history(self, color=DEFAULT_COLOR, figure_name='activity_history'):

        fig, axes = plt.subplots(len(self.layers), 2, sharex=True, sharey=True)
        for ii, layer in enumerate(self.layers):
            plot_distribution_over_time(self.samples_trained[:self.ptr+1], layer.forward_pass_activity_history[:self.ptr+1], ax=axes[ii, 0])
            plot_distribution_over_time(self.samples_trained[:self.ptr+1], layer.backward_pass_activity_history[:self.ptr+1], ax=axes[ii, 1])

        # set axis limits
        for ax in axes.ravel():
            ax.set_ylim(0., 1.)

        # label axes
        axes[ 0, 0].set_title("Data phase")
        axes[ 0, 1].set_title("Model phase")
        axes[-1, 0].set_xlabel('Samples')
        axes[-1, 1].set_xlabel('Samples')
        for jj, ax in enumerate(axes[:, 0]):
            ax.set_ylabel('Layer {}\nFraction of samples ON'.format(jj))

        fig.tight_layout()

        fig.savefig(self.figure_directory + figure_name + ".pdf")
        fig.savefig(self.figure_directory + figure_name + ".svg")


    def plot_reconstructions(self, inputs, reconstructions,
                             image_shape = (10, 10), # in tiles
                             tile_shape = (28, 28), # in pixels
                             figure_name = 'reconstructions'):

        total_samples  = image_shape[0] * image_shape[1]

        fig, axes = plt.subplots(1, 2)
        for ax, arr in zip(axes, (inputs, reconstructions)):
            arr = arr[:total_samples]
            img = get_unblockedshaped(arr, tile_shape, image_shape)
            ax.imshow(img, cmap='gray', vmin=0., vmax=1.)
            ax.set_xticks([])
            ax.set_yticks([])

        axes[0].set_title('Data')
        axes[1].set_title('Reconstruction')
        fig.tight_layout()

        fig.savefig(self.figure_directory + figure_name + ".pdf")
        fig.savefig(self.figure_directory + figure_name + ".svg")


def plot_distribution_over_time(time, values, percentiles=[0, 5, 25, 75, 95, 100], color=DEFAULT_COLOR, ax=None):
    """
    Plot a distribution of values over time.
    Arguments:
    ----------
    time -- (total time points, ) ndarray
    values -- (total time points, total values) ndarray, or list equivalent
    percentiles -- (2 * total levels, ) iterable
    """

    if ax is None:
        fig, ax = plt.subplots(1,1)

    percentile_values = np.array([np.nanpercentile(v, percentiles) for v in values])
    total_levels = len(percentiles)/2
    for ii in range(total_levels):
        ax.fill_between(time, percentile_values[:, ii], percentile_values[:, len(percentiles)-ii-1],
                        alpha = 1. / (total_levels + 1),
                        color = color)

    median = np.nanmedian(values, axis=1)
    ax.plot(time, median, color=color)


def test_plot_distribution_over_time():
    total_time_points = 10
    total_values = 1000
    values = np.random.randn(total_time_points, total_values)
    # increase variance over time
    values *= (np.arange(total_time_points)[:, None] + 1)
    # shift mean upwards over time
    values += np.arange(total_time_points)[:, None]
    # plot
    plot_distribution_over_time(np.arange(total_time_points), values)
    plt.show()


if __name__ == '__main__':

    # --------------------------------------------------------------------------------
    # load and preprocess data

    ddir = '../data/mldata'

    inputs_train, labels_train = load_mnist(ddir, 'train')
    inputs_test,  _            = load_mnist(ddir, 'test')

    inputs_train = rescale(inputs_train, 0., 1.)
    inputs_test  = rescale(inputs_test,  0., 1.)

    batch_size = 10
    inputs_train  = make_balanced_batches(inputs_train, labels_train, batch_size)

    total_epochs = 10
    test_at = np.r_[[0, 10, 30, 60, 100, 300, 600, 1000, 3000], np.arange(1, total_epochs+1) * 6000] # in batches

    # --------------------------------------------------------------------------------
    # import experiments

    # from experiments_figure_1 import experiments as experiments_figure_1
    # from experiments_figure_2 import experiments as experiments_figure_2
    # from experiments_figure_3 import experiments as experiments_figure_3
    # experiments = experiments_figure_1 + \
    #               experiments_figure_2 + \
    #               experiments_figure_3

    from experiments_figure_4 import experiments

    # --------------------------------------------------------------------------------
    # run experiments and plot outputs

    # initialise figure
    fig, ax = plt.subplots(1,1)
    fig2, ax2 = plt.subplots(1,1)

    # loop over experiments and populate plot
    for ii, (model, init_params, train_params, color, label, fname) in enumerate(experiments):

        # test_params = dict(loss_function=get_mean_squared_error,
        #                    plot_function = DiagnosticPlotter(
        #                        samples_trained=test_at,
        #                        layers=init_params['layers'],
        #                        color=color,
        #                        figure_directory='../figures/tmp/'+fname+'_'))
        test_params = dict(loss_function=get_mean_squared_error)
        plotter = partial(DiagnosticPlotter,
                          samples_trained=test_at,
                          layers=init_params['layers'],
                          color=color,
                          figure_directory='../figures/tmp/'+fname+'_')


        loss, samples = characterise_model(model             = model,
                                           init_params       = init_params,
                                           train_params      = train_params,
                                           test_params       = test_params,
                                           inputs_train      = inputs_train,
                                           inputs_test       = inputs_test,
                                           test_at           = test_at,
                                           plotter           = plotter,
                                           total_repetitions = 3)

        np.savez('../data/results_' + fname,
                 loss         = loss,
                 samples      = samples,
                 model        = str(type(model)),
                 train_params = train_params,
        )

        # --------------------------------------------------------------------------------
        # plot loss versus time

        if np.any(samples==0):
            # break the line at whenever we return to x = 0
            samples = samples.astype(np.float)
            samples[samples==0] = np.nan
            # show first data point if x-axis is log-scaled
            if samples[0] == 0:
                samples[0] = 1

        ax.errorbar(x=samples, y=loss.mean(axis=0),
                    yerr=loss.std(axis=0), errorevery=1,
                    color=color, ecolor=color, alpha=0.9,
                    linewidth=2, linestyle='-',
                    label=label)

        # annotate final loss
        final_sample = samples[-1]
        final_loss = loss[:,-1]
        ax.annotate(s='{:.3f}'.format(final_loss.mean()),
                    xy=(final_sample, final_loss.mean()), xycoords='data',
                    xytext=(5, 0), textcoords='offset points',
                    fontsize='xx-small')

        # layout plot
        ax.set_xscale('log')
        # ax.set_yscale('log')
        ax.set_ylim(0, 0.55)
        ax.set_xlabel('Training samples')
        ax.set_ylabel('Mean squared error')
        ax.legend(loc=1, fontsize='xx-small')

        fig.tight_layout()
        fig.savefig('../figures/tmp/performance_vs_time.pdf', dpi=300)
        fig.savefig('../figures/tmp/performance_vs_time.svg', dpi=300)

        # --------------------------------------------------------------------------------
        # plot comparison of final loss

        ax2.bar(ii, final_loss.mean(), yerr=final_loss.std(), color=color, alpha=0.9)

        ax2.set_xticks(range(len(experiments)))
        ax2.set_xticklabels([label for (model, init_params, train_params, color, label, fname) in experiments], rotation=90)
        ax2.set_ylabel('Mean squared error')

        fig2.tight_layout()
        fig2.savefig('../figures/tmp/final_performance.pdf', dpi=300)
        fig2.savefig('../figures/tmp/final_performance.svg', dpi=300)

    import ipdb; ipdb.set_trace()
