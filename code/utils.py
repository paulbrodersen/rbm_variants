#!/usr/bin/env python

import numpy as np
import time

from sys import stdout


def get_mean_squared_error(x, y, axis=-1):
    return np.mean((x.astype(np.float) - y.astype(np.float))**2, axis)


def get_cosine_similarity(x, y, axis=-1):
    x = x.astype(np.float)
    y = y.astype(np.float)
    numerator = np.nansum(x * y, axis=axis)
    denominator = np.nansum(np.abs(x), axis=axis) * np.nansum(np.abs(y), axis=axis)
    return numerator / denominator


def get_unblockedshaped(arr, tile_shape=None, image_shape=None):
    """
    Arguments:
    ----------
    arr: (N, M) ndarray

    tile_shape: 2-tuple (int x, int y) or None (default None)
        size of each tile in the tiled image;
        product of dimensions must match second dimension of input arrays (M = x * y);
        if None, x = y = \sqrt(M);

    image_shape: 2-tuple (int w, int h) or None (default None)
        size of image in tiles; N = w * h
        product of dimensions must match first dimension of input arrays (M = x * y);
        if None, w = h = \sqrt(N);

    Returns:
    arr_reshaped: (x*w, y*h) ndarray

    """

    n, m = arr.shape

    if tile_shape:
        x, y = tile_shape
    else:
        x = int(np.sqrt(m))
        y = int(np.sqrt(m))

    if image_shape:
        w, h = image_shape
    else:
        w = int(np.sqrt(n))
        h = int(np.sqrt(n))

    reshaped = _unblockshaped(arr.reshape(n, x, y), h*y, w*x)

    return reshaped


def _unblockshaped(arr, h, w):
    """
    Return an array of shape (h, w) where
    h * w = arr.size

    If arr is of shape (n, nrows, ncols), n sublocks of shape (nrows, ncols),
    then the returned array preserves the "physical" layout of the sublocks.
    """
    n, nrows, ncols = arr.shape
    return (arr.reshape(h//nrows, -1, nrows, ncols)
               .swapaxes(1,2)
               .reshape(h, w))


def rescale(arr, minimum=0., maximum=1.):
    # squeeze values into 0-1 range
    arr = arr.astype(np.float)
    arr -= arr.min()
    arr /= arr.max()

    # rescale
    arr *= maximum - minimum
    arr += minimum

    return arr


def make_batches(arr, batch_size=100):
    # figure out maximum number of possible batches
    total_samples, total_features = arr.shape
    total_batches = int(total_samples / batch_size)

    # trim off samples exceeding total batches * batch size such that reshape works
    arr  = arr[:total_batches*batch_size]

    # reshape
    arr  = arr.reshape((total_batches, batch_size, arr.shape[-1]))

    return arr


def make_balanced_batches(arr, labels, batch_size=100):

    # assert that it is possible to make a balanced batches
    unique_labels, inverse = np.unique(labels, return_inverse=True)
    total_labels = len(unique_labels)
    assert batch_size % total_labels == 0, \
        "Cannot create balanced batches given {} unique labels and a batch size of {}".format(total_labels, batch_size)
    repeats_per_batch = int(batch_size / total_labels)

    # figure out maximum number of possible batches
    counts = np.bincount(inverse)
    total_count_rarest_label = np.min(counts)
    total_batches = int(total_count_rarest_label / (batch_size * repeats_per_batch))

    # sort indices into batches
    order = np.argsort(labels)
    indices = np.zeros((total_labels, total_count_rarest_label), dtype=np.int)
    for ii, label in enumerate(unique_labels):
        indices[ii] = order[labels == label][:total_count_rarest_label]

    # create batches
    _, total_features = arr.shape
    new_arr = np.zeros((total_batches, batch_size, total_features))
    for ii in range(total_batches):
        new_arr[ii] = arr[indices[:, repeats_per_batch * ii : repeats_per_batch * (ii+1)].ravel()].reshape(batch_size, total_features)

    return new_arr


def shuffle_samples_across_batches(arr):
    """
    Shuffle samples across batches.

    Arguments:
    ---------
    arr -- (total batches, total samples, total features) ndarray.

    Returns:
    --------
    shuffled -- (total batches, total samples, total features) ndarray.

    """

    total_batches, total_samples, total_features = arr.shape
    shuffled = np.zeros_like(arr)
    for ii in range(total_samples):
        shuffled[:,ii,:] = _shuffle(arr[:,ii,:])

    return shuffled


def _shuffle(arr):
    idx = np.arange(len(arr))
    np.random.shuffle(idx)
    arr = arr[idx]
    return arr


# def characterise_model(model, init_params,
#                        train_params, test_params,
#                        inputs_train, inputs_test,
#                        test_at,
#                        total_batches=None,
#                        total_repetitions=3,
#                        return_anns=False):

#     total_train_batches, batch_size, total_input_features = inputs_train.shape

#     if total_batches is None:
#         total_batches = total_train_batches

#     total_tests = len(test_at)
#     error   = np.full((total_repetitions, total_tests), np.nan) # output array
#     loss    = np.full_like(error, np.nan)
#     # samples = np.arange(0, total_batches*batch_size+1, batch_size*test_every)
#     samples = batch_size * test_at
#     anns    = []

#     for rep in range(total_repetitions):

#         tic = time.time()

#         ann = model(**init_params)

#         if rep == 0: # print model specifications
#             print
#             print('--------------------------------------------------------------------------------')
#             print(ann)
#             print
#             for key, value in init_params.items():
#                 print("  {}: {}".format(key, value))
#             for key, value in train_params.items():
#                 print("  {}: {}".format(key, value))
#             print

#         inputs_train  = shuffle(inputs_train.reshape(-1, total_input_features))
#         inputs_train  = make_batches(inputs_train, batch_size)

#         total_batches_trained = 0
#         for ii in range(total_tests):
#             if test_at[ii] - total_batches_trained > 0:
#                 indices = np.arange(total_batches_trained, test_at[ii])
#                 super_batch = np.take(inputs_train, indices=indices, axis=0, mode='wrap')
#                 ann.train(super_batch, **train_params)
#                 total_batches_trained = test_at[ii]

#             loss[rep,ii] = ann.test(inputs_test, **test_params)

#             stdout.write('\r{:5d} of {:5d} total batches;'.format(rep*total_batches+test_at[ii], total_batches*total_repetitions))
#             stdout.write(' repetition: {:2d}; batch: {:3d};'.format(rep+1, test_at[ii]))
#             stdout.write(' loss: {:.3f}'.format(loss[rep,ii]))
#             stdout.flush()

#         stdout.write('\n')

#         toc = time.time()
#         stdout.write('Time elapsed: {:.1f} s\n'.format(toc-tic))

#         stdout.flush()

#         anns.append(ann)

#     if return_anns:
#         return loss, samples, anns
#     else:
#         return loss, samples

def characterise_model(model, init_params, train_params, test_params,
                       inputs_train, inputs_test,
                       test_at           = None,
                       total_repetitions = 3,
                       return_anns       = False):

    total_batches, batch_size, total_input_features = inputs_train.shape

    if test_at is None:
        test_at = np.arange(total_batches)

    total_layers = len(init_params['layers'])
    loss   = np.full((total_repetitions, len(test_at) * (total_layers -1)), np.nan) # output array
    samples = np.tile(batch_size * test_at, (total_layers -1))
    anns    = []

    for rep in range(total_repetitions):

        tic = time.time()

        ann = model(**init_params)

        if rep == 0: # print model specifications
            print
            print('--------------------------------------------------------------------------------')
            print(ann)
            print
            for key, value in init_params.items():
                print("  {}: {}".format(key, value))
            for key, value in train_params.items():
                print("  {}: {}".format(key, value))
            print

        print("Repetition {} / {}".format(rep+1, total_repetitions))

        inputs_train = shuffle_samples_across_batches(inputs_train)

        loss[rep] = ann.train(inputs_train, inputs_test, test_at, test_params, **train_params)

        stdout.write('\n')

        toc = time.time()
        stdout.write('Time elapsed: {:.1f} s\n'.format(toc-tic))

        stdout.flush()

        anns.append(ann)

    if return_anns:
        return loss, samples, anns
    else:
        return loss, samples


def subplots(nrows=1, ncols=1, *args, **kwargs):
    """
    Make plt.subplots return an array of axes even is there is only one axis.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(nrows, ncols, *args, **kwargs)
    if nrows==1 and ncols==1:
        axes = np.array([axes])
    return fig, axes
