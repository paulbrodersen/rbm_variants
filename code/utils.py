#!/usr/bin/env python

import numpy as np

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


def shuffle(arr):
    idx = np.arange(len(arr))
    np.random.shuffle(idx)
    arr = arr[idx]
    return arr


def characterise_model(model, init_params,
                       train_params, test_params,
                       inputs_train, inputs_test,
                       test_every=10,
                       total_batches=None,
                       total_repetitions=3,
                       return_anns=False):

    total_train_batches, batch_size, total_input_features = inputs_train.shape

    if total_batches is None:
        total_batches = total_train_batches

    total_tests = np.int(total_batches/test_every) + 1
    error   = np.full((total_repetitions, total_tests), np.nan) # output array
    loss    = np.full_like(error, np.nan)
    samples = np.arange(0, total_batches*batch_size+1, batch_size*test_every)
    anns    = []

    for rep in range(total_repetitions):

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

        inputs_train  = shuffle(inputs_train.reshape(-1, total_input_features))
        inputs_train  = make_batches(inputs_train, batch_size)

        # for ii in range(total_batches):
        #     if (ii % test_every) == 0:
        #         jj = int(ii/test_every)
        #         loss[rep,jj]= ann.test(inputs_test)
        #         stdout.write('\r{:5d} of {:5d} total batches;'.format(rep*total_batches+ii, total_batches*total_repetitions))
        #         stdout.write(' repetition: {:2d}; batch: {:3d};'.format(rep+1, ii))
        #         stdout.write(' loss: {:.3f}'.format(loss[rep,jj]))
        #         stdout.flush()
        #     ann.train(inputs_train[ii%total_train_batches], **train_params)

        for ii in range(int(total_batches / test_every)):
            loss[rep,ii] = ann.test(inputs_test, **test_params)
            stdout.write('\r{:5d} of {:5d} total batches;'.format(rep*total_batches+ii*test_every, total_batches*total_repetitions))
            stdout.write(' repetition: {:2d}; batch: {:3d};'.format(rep+1, ii*test_every))
            stdout.write(' loss: {:.3f}'.format(loss[rep,ii]))
            stdout.flush()

            indices = np.arange(ii * test_every, (ii+1) * test_every)
            super_batch = np.take(inputs_train, indices=indices, axis=0, mode='wrap')
            ann.train(super_batch, **train_params)

        ii += 1
        loss[rep,ii] = ann.test(inputs_test, **test_params)
        stdout.write('\r{:5d} of {:5d} total batches;'.format(rep*total_batches+ii*test_every, total_batches*total_repetitions))
        stdout.write(' repetition: {:2d}; batch: {:3d};'.format(rep+1, ii*test_every))
        stdout.write(' loss: {:.3f}\n'.format(loss[rep,ii]))
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
