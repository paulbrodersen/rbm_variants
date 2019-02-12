#!/usr/bin/env python

"""
Experiments to produce panels for Fig. 1 in Brodersen et al (2019).

These simulations show that RBMs with independent and asymmetric forward
and backward weights learn to reconstruct inputs.
"""

from rbm_variants import (LogisticLayer, BoltzmannLayer,
                          RestrictedBoltzmannMachine, DirectedRBM,
)

# network layout
network_layout = dict(layers=[LogisticLayer(784), BoltzmannLayer(400)])

# parameters
experiments = []
experiments.append([RestrictedBoltzmannMachine, dict(scale_weights_by=0.1,                                        **network_layout), dict(cd=3, eta=0.01),                                              '#1f77b4',    'Canonical RBM',                                  'f1_canonical_rbm'])
experiments.append([DirectedRBM,                dict(scale_forward_weights_by=0.1, scale_backward_weights_by=0.1, **network_layout), dict(cd=3, eta=0.01, update_forward=True,  update_backward=True ), '#9467bd',    'Directed RBM',                                   'f1_directed_rbm'])
