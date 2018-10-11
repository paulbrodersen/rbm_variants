#!/usr/bin/env python

"""
Experiments to roduce panels for Fig. 2 in Brodersen et al (2018).

These plots show that uni-directional connections are sufficient for learning.
"""

from rbm_variants import (LogisticLayer, BoltzmannLayer,
                          SparseRBM, SparseDirectedRBM, ComplementaryRBM,
)

# network layout
network_layout = dict(layers=[LogisticLayer(784), BoltzmannLayer(400)])

# parameters
experiments = []
experiments.append([SparseRBM,         dict(connection_probability=0.5, scale_weights_by=0.1,                                        **network_layout), dict(cd=3, eta=0.01), '#1f77b4', 'Sparse RBM \n($p=0.5$)', 'f2_sparse_rbm'])
experiments.append([SparseDirectedRBM, dict(connection_probability=0.5, scale_forward_weights_by=0.1, scale_backward_weights_by=0.1, **network_layout), dict(cd=3, eta=0.01), '#9467bd', 'Sparse, directed RBM \n($p=0.5$)', 'f2_sparse_directed_rbm'])
experiments.append([ComplementaryRBM,  dict(connection_probability=0.5, scale_forward_weights_by=0.1, scale_backward_weights_by=0.1, **network_layout), dict(cd=3, eta=0.01), '#d62728', 'Sparse directed RBM \nw/o bidirectional connections \n($p=0.5$)', 'f2_complementary_rbm'])
