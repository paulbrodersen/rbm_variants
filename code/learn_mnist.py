#!/usr/bin/env python

"""
Train RBM variants to reconstruct MNIST digits
"""

import numpy as np
import matplotlib.pyplot as plt
import mnist

from functools import partial
from rbm_variants import BoltzmannLayer, RestrictedBoltzmannMachine, DirectedRBM
from utils import rescale, make_batches, characterise_model, subplots, get_unblockedshaped, get_cosine_similarity, get_mean_squared_error

# styling
import matplotlib as mpl
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False

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


def make_diagnostic_plots(forward_pass_activities, backward_pass_activities, layers, color='#1f77b4', fdir='../figures/'):
    total_layers = len(layers)

    # --------------------------------------------------------------------------------
    # activity profile

    fig, axes = plt.subplots(total_layers, 2, sharex=True, sharey=True)
    for ii, activities in enumerate([forward_pass_activities, backward_pass_activities]):
        for jj, activity in enumerate(activities):
            axes[jj, ii].hist(np.mean(activity, axis=0), bins=np.linspace(0., 1., 11), color=color) # across stimuli
            axes[jj,  0].set_ylabel('Number of units (layer {})'.format(jj))

    axes[0,0].set_title("Forward pass")
    axes[0,1].set_title("Backward pass")
    axes[-1, 0].set_xlabel('Fraction of samples ON')
    axes[-1, 1].set_xlabel('Fraction of samples ON')

    for ax in axes.ravel():
        ax.set_xlim(0., 1.)
        ax.set_ylim(0, 500)

    fig.tight_layout()
    fig.savefig(fdir + "activity_distribution.pdf")
    fig.savefig(fdir + "activity_distribution.svg")

    # --------------------------------------------------------------------------------
    # weight distribution

    fig, axes = plt.subplots(total_layers-1, 2, sharex=True, sharey=True)
    for ii in range(total_layers-1):

        if total_layers-1 == 1:
            ax1, ax2 = axes
        else:
            ax1, ax2 = axes[ii]

        if ii == 0:
            ax1.set_title("Forward weights")
            ax2.set_title("Backward weights")
        ax1.set_ylabel("Number of weights from layer {} to layer {}".format(ii, ii+1))

        # ax1.hist(layers[ii].forward_weights.ravel(), bins=np.linspace(-5, 5, 11), color=color)
        # ax2.hist(layers[ii+1].backward_weights.ravel(), bins=np.linspace(-5, 5, 11), color=color)

        ax1.hist(layers[ii].forward_weights.ravel(), bins=np.linspace(-10, 10, 21), color=color)
        ax2.hist(layers[ii+1].backward_weights.ravel(), bins=np.linspace(-10, 10, 21), color=color)

    ax1.set_xlabel('Weight value [AU]')
    ax2.set_xlabel('Weight value [AU]')

    for ax in axes.ravel():
        # ax.set_xlim(-10, 10)
        ax.set_xlim(-5, 5)
        ax.set_ylim(0, 120 * 1e+3)

    fig.tight_layout()
    fig.savefig(fdir + "weight_distribution.pdf")
    fig.savefig(fdir + "weight_distribution.svg")

    # --------------------------------------------------------------------------------
    # weight alignment

    fig, axes = plt.subplots(total_layers-1, 1)
    for ii in range(total_layers-1):

        if total_layers-1 == 1:
            ax = axes
        else:
            ax = axes[ii]

        ax.plot(layers[ii].forward_weights.ravel(), layers[ii+1].backward_weights.transpose().ravel(), '.', markersize=1, color=color, alpha=0.1)

    ax.set_xlabel('Forward weight value [AU]')
    ax.set_ylabel('Backward weight value [AU]')

    fig.tight_layout()
    # fig.savefig(fdir + "weight_alignment.pdf")
    fig.savefig(fdir + "weight_alignment.png")
    fig.savefig(fdir + "weight_alignment.svg")

    # --------------------------------------------------------------------------------
    # biases

    extremum = 15
    fig, axes = plt.subplots(total_layers, sharex=True, sharey=True)
    for ii in range(total_layers):
        # axes[ii].hist(layers[ii].biases, color=color)
        axes[ii].hist(layers[ii].biases, bins=np.linspace(-extremum, extremum, 31), color=color)
        axes[ii].set_ylabel('Number of biases (layer {})'.format(ii))
    axes[ii].set_xlabel('Bias value [AU]')

    for ax in axes.ravel():
        ax.set_xlim(-extremum, extremum)
        ax.set_ylim(0, 400)

    fig.tight_layout()
    fig.savefig(fdir + "bias_distribution.pdf")
    fig.savefig(fdir + "bias_distribution.svg")

    # --------------------------------------------------------------------------------
    # reconstructions

    inputs = forward_pass_activities[0]
    outputs = backward_pass_activities[0]
    image_shape = (10, 10) # in tiles
    tile_shape = (28, 28) # in pixels
    total_samples  = image_shape[0] * image_shape[1]

    fig, axes = plt.subplots(1, 2)
    for ax, arr in zip(axes, (inputs, outputs)):
        arr = arr[:total_samples]
        img = get_unblockedshaped(arr, tile_shape, image_shape)
        ax.imshow(img, cmap='gray', vmin=0., vmax=1.)
        ax.set_xticks([])
        ax.set_yticks([])

    axes[0].set_title('Data')
    axes[1].set_title('Reconstruction')

    fig.tight_layout()
    fig.savefig(fdir + "reconstructions.pdf")
    fig.savefig(fdir + "reconstructions.svg")

    # plt.ion()
    # plt.show()
    # raw_input("Press any key to close figures...")
    plt.close('all')


if __name__ == '__main__':

    # --------------------------------------------------------------------------------
    # load and preprocess data

    ddir = '../data/mldata'

    inputs_train, _ = load_mnist(ddir, 'train')
    inputs_test, _ = load_mnist(ddir, 'test')

    inputs_train = rescale(inputs_train, 0., 1.)
    inputs_test  = rescale(inputs_test,  0., 1.)

    batch_size = 100
    inputs_train  = make_batches(inputs_train, batch_size)

    total_epochs = 2
    test_at = np.r_[[0, 1, 3, 6, 10, 30, 60, 100, 300], np.arange(1, total_epochs+1) * 600] # in batches

    # --------------------------------------------------------------------------------
    # define experiments

    # test_params = dict(loss_function=get_cosine_similarity, plot_function=make_diagnostic_plots)
    # test_params = dict(loss_function=get_mean_squared_error, plot_function=make_diagnostic_plots)

    # network layout
    network_layout = dict(layers=[BoltzmannLayer(inputs_train.shape[-1]), BoltzmannLayer(400)])

    # parameters
    experiments = []
    experiments.append([RestrictedBoltzmannMachine, network_layout, dict(cd=3, eta=0.01),                                              '#1f77b4',   'Standard RBM',              'rbm'])
    experiments.append([DirectedRBM,                network_layout, dict(cd=3, eta=0.01),                                              '#9467bd',   'Directed RBM',              'directed'])
    # experiments.append([DirectedRBM,                network_layout, dict(cd=3, eta=0.01, update_forward=False, update_backward=True),  '#d62728',   r'Learn w$_{B}$ and biases', 'update_wb_and_biases'])
    # experiments.append([DirectedRBM,                network_layout, dict(cd=3, eta=0.01, update_forward=True, update_backward=False),  '#ff7f0e',   r'Learn w$_{F}$ and biases', 'update_wf_and_biases'])
    # experiments.append([DirectedRBM,                network_layout, dict(cd=3, eta=0.01, update_forward=False, update_backward=False), 'lightgray', r'Learn only biases',        'update_biases'])

    # --------------------------------------------------------------------------------
    # optimize learning rate

    # colors = ['green', 'lime', 'turquoise', 'blue', 'violet', 'red', 'orange', 'yellow']
    # etas   = [   0.001,   0.005,   0.01,        0.05,    0.1,      0.5,    1.,       5.]
    # etas = [1e-3, 1e-2, 1e-1, 1.]
    # for eta, color in zip(etas, colors):
    #     experiments.append([RestrictedBoltzmannMachine, network_layout, dict(cd=3, eta=eta),                                             color, r'Standard RBM ($\eta = {}$)'.format(eta),          'rbm'])
    #     experiments.append([DirectedRBM,                network_layout, dict(cd=3, eta=eta, update_forward=False, update_backward=True), color, r'Learn w$_{B}$ and biases ($\eta = {}$)'.format(eta), 'backward_rbm'])
    #     experiments.append([DirectedRBM,                network_layout, dict(cd=3, eta=eta, update_forward=True, update_backward=False), color, r'Learn w$_{F}$ and biases ($\eta = {}$)'.format(eta),  'forward_rbm'])

    # --------------------------------------------------------------------------------
    # run experiments and plot outputs

    # initialise figure
    fig, ax = plt.subplots(1,1)

    # loop over experiments and populate plot
    for model, init_params, train_params, color, label, fname in experiments:

        test_params = dict(loss_function=get_mean_squared_error,
                           # plot_function=None)
                           plot_function=partial(make_diagnostic_plots, color=color, fdir='../figures/'+fname+'--'))

        loss, samples = characterise_model(model             = model,
                                           init_params       = init_params,
                                           train_params      = train_params,
                                           test_params       = test_params,
                                           inputs_train      = inputs_train,
                                           inputs_test       = inputs_test,
                                           test_at           = test_at,
                                           total_batches     = test_at[-1],
                                           total_repetitions = 1)

        np.savez('../data/results--' + fname,
                 loss         = loss,
                 samples      = samples,
                 model        = str(type(model)),
                 train_params = train_params,
        )

        # plot loss
        samples += 1 # show first data point if x-axis is log-scaled and first test is at samples batches = 0
        ax.errorbar(x=samples, y=loss.mean(axis=0),
                    yerr=loss.std(axis=0), errorevery=1,
                    color=color, ecolor=color, alpha=0.9,
                    linewidth=2, linestyle='-',
                    label=label)

        # annotate final loss
        final_sample = samples[-1]
        final_loss = loss[:,-1].mean()
        ax.annotate(s='{:.3f}'.format(final_loss),
                    xy=(final_sample, final_loss), xycoords='data',
                    xytext=(5, 0), textcoords='offset points',
                    fontsize='xx-small')

        # layout plot
        ax.set_xscale('log')
        # ax.set_yscale('log')
        ax.set_ylim(0, 0.5)
        # ax.set_ylim(0.01, 0.5)
        ax.set_xlabel('Training samples')
        ax.set_ylabel('Mean squared error')
        ax.legend(loc=1, fontsize='xx-small')

        fig.tight_layout()
        fig.savefig('../figures/performance_vs_time.pdf', dpi=300)
        fig.savefig('../figures/performance_vs_time.svg', dpi=300)

    # plt.ion()
    # plt.show()
    # raw_input('\nPress any key to close all figures.')
    # plt.close('all')
