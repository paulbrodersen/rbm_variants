#!/usr/bin/env python

"""
Experiments to produce panels for Fig. 3 in Brodersen et al (2018).

These plots show that updating the backward weights is sufficient for learning.
"""

from rbm_variants import (LogisticLayer, BoltzmannLayer,
                          RestrictedBoltzmannMachine, DirectedRBM,
)

# network layout
network_layout = dict(layers=[LogisticLayer(784), BoltzmannLayer(400)])

# parameters
experiments = []
experiments.append([RestrictedBoltzmannMachine, dict(scale_weights_by=0.1,                                        **network_layout), dict(cd=3, eta=0.01),                                              '#1f77b4',    'Canonical RBM',                                  'f3_canonical_rbm'])
experiments.append([DirectedRBM,                dict(scale_forward_weights_by=0.1, scale_backward_weights_by=0.1, **network_layout), dict(cd=3, eta=0.01, update_forward=True,  update_backward=True ), '#9467bd',    'Directed RBM',                                   'f3_directed_rbm'])
experiments.append([DirectedRBM,                dict(scale_forward_weights_by=1.0, scale_backward_weights_by=0.1, **network_layout), dict(cd=3, eta=0.01, update_forward=False, update_backward=True ), '#d62728',   r'Directed RBM with constant w$_{F}$',             'f3_directed_rbm_with_constant_w_F'])
experiments.append([DirectedRBM,                dict(scale_forward_weights_by=0.1, scale_backward_weights_by=1.0, **network_layout), dict(cd=3, eta=0.01, update_forward=True,  update_backward=False), '#ff7f0e',   r'Directed RBM with constant w$_{B}$',             'f3_directed_rbm_with_constant_w_B'])
experiments.append([DirectedRBM,                dict(scale_forward_weights_by=0.1, scale_backward_weights_by=1.0, **network_layout), dict(cd=3, eta=0.01, update_forward=False, update_backward=False), 'lightgray', r'Directed RBM with constant w$_{F}$ and w$_{B}$', 'f3_directed_rbm_with_constant_w_F_and_w_B'])
