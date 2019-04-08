#!/usr/bin/env python

"""
Experiments to produce panels for Fig. 4 in Brodersen et al (2019).

These experiments show that directed RBMs also learn when stacked.

TODO:
- add comparison to 2-layer network

"""

import matplotlib.pyplot as plt

from rbm_variants import (LogisticLayer, BoltzmannLayer,
                          RestrictedBoltzmannMachine, DirectedRBM,
)

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

# network layout
# network_layout = dict(layers=[LogisticLayer(784), BoltzmannLayer(1000)])
network_layout = dict(layers=[LogisticLayer(784), BoltzmannLayer(1000), BoltzmannLayer(1000)])

# parameters
experiments = []

# for ii, s in enumerate([0.1, 0.5, 1., 5., 10.]):
#     experiments.append([RestrictedBoltzmannMachine,
#                         dict(scale_weights_by=s, **network_layout),
#                         dict(cd=3, eta=0.01),
#                         colors[ii], '$|w| = {}$'.format(s), 'test_{}'.format(ii)])

# for ii, v in enumerate([0.001, 0.005, 0.01, 0.05, 0.1, 0.5]):
for ii, v in enumerate([0.1, 0.5]):
    experiments.append([RestrictedBoltzmannMachine,
                        dict(scale_weights_by=1., **network_layout),
                        dict(cd=3, eta=v),
                        colors[ii], '$\eta = {}$'.format(v), 'test_{}'.format(ii)])

# experiments.append([RestrictedBoltzmannMachine, dict(scale_weights_by=1.,                                        **network_layout), dict(cd=3, eta=0.01),                                              '#1f77b4',    'Canonical RBM',                                  'f4_canonical_rbm'])
# experiments.append([DirectedRBM,                dict(scale_forward_weights_by=0.1, scale_backward_weights_by=0.1, **network_layout), dict(cd=3, eta=0.01, update_forward=True,  update_backward=True ), '#9467bd',    'Directed RBM',                                   'f4_directed_rbm'])
