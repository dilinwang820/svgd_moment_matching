import sys
import numpy as np

import theano
import theano.tensor as T
from updates import Adagrad
from tqdm import tqdm
from rng import t_rng, np_rng
from theano_utils import floatX, sharedX
from math import pi

def median_distance(H, e=1e-6, log_ratio_bandwidth=False):
    if H.ndim != 2:
        raise NotImplementedError

    V = H.flatten()
    # median distance
    h = T.switch(T.eq((V.shape[0] % 2), 0),
        # if even vector
        T.mean(T.sort(V)[ ((V.shape[0] // 2) - 1) : ((V.shape[0] // 2) + 1) ]),
        # if odd vector
        T.sort(V)[V.shape[0] // 2])

    if log_ratio_bandwidth:
        h /= T.log(H.shape[0].astype('float32') + 1.0)
    return h


def sqr_dist(x, y, e=1e-8):
    if x.ndim != 2:
        raise NotImplementedError
    xx = T.sqr(T.sqrt((x*x).sum(axis=1) + e))
    yy = T.sqr(T.sqrt((y*y).sum(axis=1) + e))
    dist = T.dot(x, y.T)
    dist *= -2.
    dist += xx.dimshuffle(0, 'x')
    dist += yy.dimshuffle('x', 0)
    return dist


def rbf_kernel(x, score_q, **model_params):

    H = sqr_dist(x, x)
    h = median_distance(H)

    kxy = T.exp(-H / h)
    dxkxy = -T.dot(kxy, x)
    sumkxy = T.sum(kxy, axis=1).dimshuffle(0, 'x')
    dxkxy = (dxkxy + x * sumkxy) * 2. / h

    return kxy, dxkxy


def random_features_kernel(x, n_features, score_q, fixed_weights=True, cos_feat_dim_scale=True, **model_params):
    dim = x.get_value().shape[1]

    H = sqr_dist(x, x)
    h = median_distance(H)
    #h = select_h_by_ksdv(x, score_q, **model_params)
    gamma = 1./h

    if fixed_weights:
        random_offset = sharedX(np_rng.uniform(0, 2*pi, (n_features,)))
        weights = sharedX(np_rng.normal(0, 1, (dim, n_features)))
        random_weights = T.sqrt(2*gamma) * weights
    else:
        #random_weights = T.sqrt(2 * gamma) * t_rng.normal(
        #     (x.shape[1], n_features))

        #random_offset = t_rng.uniform((n_features,), 0, 2 * pi)
        raise NotImplementedError

    if cos_feat_dim_scale:
        alpha = T.sqrt(2.) / T.sqrt(n_features).astype(theano.config.floatX)
    else:
        alpha = T.sqrt(2.)

    coff = T.dot(x, random_weights) + random_offset
    projection = alpha * T.cos(coff)
    
    kxy = T.dot(projection, projection.T)

    sinf = -alpha*T.sin(coff)
    wd = random_weights.T

    inner = T.sum(sinf, axis=0).dimshuffle(0,'x') * wd
    dxkxy = T.dot(projection, inner)

    return kxy, dxkxy


def poly_kernel(x, substract_mean=False, with_scale=False, e=1e-8):
    if substract_mean:
        x = x - T.mean(x, axis=0)

    kxy = 1 + T.dot(x, x.T)
    dxkxy = x * x.shape[0].astype(theano.config.floatX)

    if with_scale:
        dim = x.shape[1].astype(theano.config.floatX)
        kxy /= (dim+1)
        dxkxy /= (dim+1)
    return kxy, dxkxy


def svgd_gradient(x, score_q, kernel='rbf', n_features=-1, fixed_weights=True, **model_params):

    dim = x.get_value().shape[1]
    grad = score_q(x, **model_params)

    if kernel == 'random_feature':
        assert n_features > 0, 'illegal inputs'
        kxy, dxkxy = random_features_kernel(x, n_features, score_q, fixed_weights, **model_params)

    elif kernel == 'rbf':
            kxy, dxkxy = rbf_kernel(x, score_q, **model_params)

    elif kernel == 'poly':
            kxy, dxkxy = poly_kernel(x, substract_mean=True)
    
    elif kernel == 'combine':
        assert n_features > 0, 'illegal inputs'
        k_lin, dk_lin = poly_kernel(x, substract_mean=False, with_scale=True)
        if n_features > dim+1:
            k_cos, dk_cos = random_features_kernel(x, n_features-dim-1, score_q, fixed_weights, **model_params)
        alpha = 0.5
        kxy = alpha * k_lin + (1. - alpha) * k_cos
        dxkxy = alpha * dk_lin + (1. - alpha) * dk_cos

    svgd_grad = (T.dot(kxy, grad) + dxkxy) / T.sum(kxy, axis=1, keepdims=True)
    return svgd_grad


'''
    https://www.ics.uci.edu/~welling/publications/papers/stoclangevin_v6.pdf
'''
def langevin(x0, score_q, lr=1e-2, max_iter=500, progressbar=True, trace=False, **model_params):

    theta = theano.shared(x0)
    i = theano.shared(floatX(0))

    stepsize = T.cast(lr * (i+1)**(-0.55), theano.config.floatX)
    grad = score_q(theta, **model_params)
    update = stepsize * grad/2. + T.sqrt(stepsize) * t_rng.normal(size=theta.shape)

    cov_grad = T.sum(update**2, axis=1).mean()

    langevin_step = theano.function([], [], updates=[(theta, theta+update), (i, i+1)])

    if progressbar:
        progress = tqdm(np.arange(max_iter))
    else:
        progress = np.arange(max_iter)

    xx = []
    for _ in progress:
        langevin_step()
        if trace:
            xx.append(theta.get_value())

    theta_val = theta.get_value()
    return theta_val, xx


def svgd(x0, score_q, max_iter=2000, kernel='rbf', n_features=-1, fixed_weights=True, optimizer=None, progressbar=True, trace=False, **model_params):

    theta = theano.shared(floatX(x0))
    epsilon = theano.shared(floatX(np.zeros(x0.shape[1])))

    svgd_grad = svgd_gradient(theta, score_q, kernel, n_features, fixed_weights, **model_params)

    # Initialize optimizer
    if optimizer is None:
        optimizer = Adagrad(lr=1e-3, alpha=.5)  # TODO. works better with regularizer for high dimension data  

    svgd_updates = optimizer([theta], [-1 * svgd_grad])

    _svgd_step = theano.function([], [], updates=svgd_updates)

    # Run svgd optimization
    if progressbar:
        progress = tqdm(np.arange(max_iter))
    else:
        progress = np.arange(max_iter)

    xx, grad_err = [], []

    for iter in progress:
        _svgd_step()
        if trace:
            xx.append(theta.get_value())

    theta_val = theta.get_value()

    return theta_val, xx

