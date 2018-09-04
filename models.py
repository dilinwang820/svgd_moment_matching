import theano
import theano.tensor as T
import numpy as np


def sqr(x, y, e=1e-8):
    if x.ndim != 2 or y.ndim !=2:
        raise NotImplementedError

    xx = T.sqr(T.sqrt((x*x).sum(axis=1) + e))
    yy = T.sqr(T.sqrt((y*y).sum(axis=1) + e))
    dist = T.dot(x, y.T)
    dist *= -2
    dist += xx.dimshuffle(0, 'x')
    dist += yy.dimshuffle('x', 0)

    return dist


def score_gaussian(x, mu, A):   # A, inverse of covariance matrix
    return -T.dot(x-mu.dimshuffle('x', 0), A).astype(theano.config.floatX)


def logp_gaussian(x, mu, A):
    x = x - mu
    #x = T.reshape(x, (x.shape[0], 1, -1))
    #A = T.tile(A, (x.shape[0], 1, 1))
    #logpdf = -.5*T.batched_dot(T.batched_dot(x, A), x.reshape((x.shape[0], -1, 1))).flatten()
    logpdf = -0.5 * T.sum(T.dot(x, A) * x, axis=1)
    return logpdf


def score_gmm(X, mu, std, weights):
    if weights is None:
        weights = T.ones(mu.shape[0])

    weigths = weights / T.sum(weights)
        
    def posterior(X, mu, std, weights):
        xmu = (-sqr(X, mu) / 2. / std ** 2) + T.log(weights).dimshuffle('x', 0)
        prob = T.exp(xmu - T.argmax(xmu, axis=1).dimshuffle(0, 'x'))
        prob = prob / T.sum(prob, axis=1).dimshuffle(0, 'x')
        return prob
    
    nc, _ = mu.shape
    nx, d = X.shape
    
    diff = -1 * (T.tile(X.reshape((1, nx, d)), (nc, 1, 1)) - mu.reshape((nc, 1, d))) / (std ** 2)
    P = posterior(X, mu, std, weigths)
    score = T.batched_dot(P.reshape((nx, 1, nc)), diff.dimshuffle([1, 0, 2])).reshape((nx, d))
    
    return score.astype(theano.config.floatX)


def logp_gmm(X, mu, std, weights=None):
    if weights is None:
        weights = T.ones(mu.shape[0])
        
    weigths = weights / T.sum(weights)

    xmu = (-sqr(X, mu) / 2. / std ** 2) + T.log(weights)
    xmax = T.argmax(xmu, axis=1, keepdims=True)
    prob = T.exp(xmu - xmax)
    logsum = T.log(T.sum(prob, axis=1, keepdims=True)) + xmax

    return logsum


