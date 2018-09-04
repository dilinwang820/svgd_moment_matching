import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist
import theano
import theano.tensor as T
from theano_utils import floatX, sharedX
from ops import sqr_dist, median_distance

def comm_func_eval(samples, ground_truth, h0, score_q=None, **model_params):

    def ex():
        f0 = np.mean(samples, axis=0)
        f1 = np.mean(ground_truth, axis=0)
        return np.mean((f0-f1)**2)

    def exsqr():
        f0 = np.mean(samples**2, axis=0)
        f1 = np.mean(ground_truth**2, axis=0)
        return np.mean((f0-f1)**2)

    out = {}
    out['ex'] = ex()
    out['exsqr'] = exsqr()

    if score_q is not None:
        ksd_u, ksd_v = ksd_eval(samples, h0, score_q, **model_params)
        out['ksd_u'] = ksd_u.sum()
        out['ksd_v'] = ksd_v.sum()

    ## mmd
    out['mmd'] = mmd(samples, ground_truth, h0)[0]
    return out


def ksd_eval(X0, h0, score_q, **model_params):

    X = sharedX(X0)
    h = sharedX(h0)

    Sqx = score_q(X, **model_params)

    H = sqr_dist(X, X)
    h = T.sqrt(h/2.)

    V = H.flatten()
    # median distance
    h = T.switch(T.eq((V.shape[0] % 2), 0),
            # if even vector
            T.mean(T.sort(V)[((V.shape[0] / 2) - 1): ((V.shape[0] / 2) + 1)]),
            # if odd vector
            T.sort(V)[V.shape[0] // 2])

    # compute the rbf kernel
    Kxy = T.exp(-H / h ** 2 / 2.)

    Sqxdy = -(T.dot(Sqx, X.T) - T.tile(T.sum(Sqx * X, axis=1).dimshuffle(0, 'x'), (1, X.shape[0]))) / ( h ** 2)

    dxSqy = T.transpose(Sqxdy)
    dxdy = (-H / (h ** 4) + X.shape[1].astype(theano.config.floatX) / (h ** 2))

    M = (T.dot(Sqx, Sqx.T) + Sqxdy + dxSqy + dxdy) * Kxy 
    M2 = M - T.diag(T.diag(M)) 

    ksd_u = T.sum(M2) / (X.shape[0] * (X.shape[0] - 1)) 
    ksd_v = T.sum(M) / (X.shape[0] ** 2) 

    f = theano.function(inputs=[], outputs=[ksd_u, ksd_v])

    return f()


### A Kernel Method for the Two Sample Problem
def mmd(x0, y0, h0 = None):

    assert h0 is not None, 'illegal inputs'
    x = sharedX(np.copy(x0))
    h = sharedX(h0)

    if len(y0) > 5000:
        y = sharedX(np.copy(y0[:5000]))
    else:
        y = sharedX(np.copy(y0))

    kxx = sqr_dist(x, x)
    kxy = sqr_dist(x, y)
    kyy = sqr_dist(y, y)

    kxx = T.exp(-kxx / h)
    kxy = T.exp(-kxy / h)
    kyy = T.exp(-kyy / h)

    m = x.shape[0].astype(theano.config.floatX)
    n = y.shape[0].astype(theano.config.floatX)

    sumkxx = T.sum(kxx)
    sumkxy = T.sum(kxy)
    sumkyy = T.sum(kyy)

    mmd = T.sqrt( sumkxx / (m*m) + sumkyy / (n*n) - 2.*sumkxy / (m*n))

    f = theano.function(inputs=[], outputs=[mmd])
    return f()



