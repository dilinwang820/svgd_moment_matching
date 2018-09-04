import sys
import numpy as np

import theano
import theano.tensor as T

from updates import Adagrad
from theano_utils import floatX, sharedX
from ops import svgd
from rng import np_rng
from evaluate import comm_func_eval
from scipy.stats import wishart

from scipy.stats import multivariate_normal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models import score_gaussian, logp_gaussian
import scipy.io as sio

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--n_particles', type=int, required=False, default=200, help='number of particles')
parser.add_argument('--dim', type=int, required=False, default=100, help='Dim')
parser.add_argument('--n_iter', type=int, required=False, default=1, help='average over #n_iters')
parser.add_argument('--max_iter', type=int, required=False, default=5000, help='maximum iteration')
parser.add_argument('--cond_num', type=int, required=False, default=10, help='condition number')
opt = parser.parse_args()
assert opt.n_particles >= opt.dim, 'n_particles >= dim, illegal inputs'


def init_model(dim, Q, cond_num):

    def search_optm_cond(V, cond_num_star):
        min_c = 0
        max_c = 100

        if cond_num_star == 1:
            return 0, 1

        while True:
            curr_c = (min_c + max_c) / 2.
            M = np.eye(dim,) + curr_c * V
            eigvals = np.linalg.eigvals(M)
            assert np.all(eigvals > 0), 'illegal input'
            val = np.max(eigvals) / np.min(eigvals)
            if val - cond_num_star > 0.3:
                max_c = curr_c
            elif val - cond_num_star < -0.3:
                min_c = curr_c
            else:
                return curr_c, val

    t0 = np.eye(dim)

    mu0 = sharedX(np_rng.uniform(-3, 3, size=(dim,)))
    V = np.dot(np.dot(Q.T, np.diag(1. + np.random.uniform(0, 1, size=(dim,)))), Q)
    coeff, cond_approx = search_optm_cond(V, cond_num)
    M = np.eye(dim,) + coeff * V

    A0 = sharedX(np.linalg.inv(M))
    model_params = {'mu':mu0, 'A':A0}

    ### score function
    score_q = score_gaussian
    log_prob = logp_gaussian

    ## ground truth
    gt = np.random.multivariate_normal(mu0.get_value(), M, (10000,), check_valid='raise')

    return model_params, score_q, log_prob, gt


max_iter = opt.max_iter
n_iter = opt.n_iter
fixed_weights = True
d0 = opt.dim
cond_num = opt.cond_num
n_samples = opt.n_particles

'''
    poly: svgd with linear kernel
    random_feature: svgd with random feature kernel
    rbf: svgd with rbf kernel
    combine: combine linear kernel and random feature kernel
'''
all_algorithms = ['poly', 'random_feature', 'rbf', 'combine', 'mc']

for ii in range(1, n_iter+1):
    Q = np.random.normal(size=(d0, d0))
    #var_n_samples = np.sort(np.concatenate((np.exp(np.linspace(np.log(10), np.log(500), 10)),[d0])).astype('int32'))
    model_params, score_q, log_prob, gt0 = init_model(d0, Q, cond_num)

    from scipy.spatial.distance import cdist
    H = cdist(gt0[:1000], gt0[:1000])**2
    h0 = np.median(H.flatten())

    n_features = n_samples
    x0 = floatX(np.random.uniform(-5, 5, [n_samples, d0]))

    for alg in all_algorithms:

        if alg == 'mc':
            xc = gt0[-n_samples:]
        else:
            optimizer = Adagrad(lr=5e-3, alpha=0.9)
            xc, _ = svgd(x0, score_q, max_iter=max_iter, kernel=alg, n_features=n_features, fixed_weights=True, optimizer=optimizer, trace=False, **model_params)

        eval_res = comm_func_eval(xc, gt0, h0, **model_params)
        print 'iter',ii, 'cond', cond_num, 'alg',alg, 'dim',d0, 'n_samples',n_samples, 'ex',eval_res['ex'], 'exsqr',eval_res['exsqr'], 'mmd',eval_res['mmd']

        sys.stdout.flush()


