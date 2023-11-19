"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture



def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    n, d = X.shape
    mu,var, p = mixture
    K = mu.shape[0]

    func = np.zeros((n,K), dtype=np.float64)

    for i in range(n):
        Cu_i = X[i,:] != 0

        dimen = np.sum(Cu_i)

        pre_exp = (-dimen/2.0)*np.log(2*np.pi*var)

        diff = X[i,Cu_i] -mu[:,Cu_i]

        norm = np.sum(diff**2, axis=1)

        func[i,:] = pre_exp -norm/(2*var)
    func = func + np.log(p + 1e-16)

    logsums = np.log(np.sum(np.exp(func),axis=1)).reshape(-1,1)

    log_p = func - logsums

    log_LH = np.sum(logsums, axis=0).item()

    return np.exp(log_p),log_LH


def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    
    K = post.shape[1]

    n_j = np.sum(post, axis=0)

    pi = n_j/n

    mu = (np.dot(post.T,X))/n_j.reshape(-1,1)

    
    norms = np.zeros((n,K),dtype=np.float64)

    for i in range(n):
        di = X[i,:] - mu
        
        norms[i,:] = np.sum(di**2, axis=1)
    
    var_r = np.sum(post*norms,axis=0)/(n_j*d)

    return GaussianMixture(mu,var_r,pi)


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    old_log_LH = None
    new_log_LH = None

    while old_log_LH is None or (new_log_LH-old_log_LH) > 1e-6*np.abs(new_log_LH):

        old_log_LH = new_log_LH

        post, new_log_LH = estep(X,mixture)

        mixture = mstep(X,post)

    return mixture, post, new_log_LH 
