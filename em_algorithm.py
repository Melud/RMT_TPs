import numpy as np
import scipy
import scipy.linalg
import scipy.spatial
from scipy.special import logsumexp
from dataclasses import dataclass
from tqdm import tqdm

gen = np.random.default_rng()

def expnormalize(log_a, axis=None):
    """
    returns, over axis,
        exp(log_a(i))/(sum_j exp(log_a(j)))
    """
    c = np.amax(log_a, axis=axis, keepdims=True)
    exp_log_a = np.exp(log_a - c)
    return exp_log_a/np.sum(exp_log_a, axis=axis, keepdims=True)
    
def log_gaussian_likelihood(x, mu, sigma):
    # returns all the log of the guaussian likelihoods
    n, d = x.shape
    m, _ = mu.shape
    assert mu.shape == (m, d)
    assert sigma.shape == (m, d, d)

    signs, dets = np.linalg.slogdet(sigma)
    assert dets.shape == (m,)
    assert signs.shape == (m,)
    assert (signs == 1).all(), (signs, dets)

    x_centered = x[:, np.newaxis, :] - mu[np.newaxis, :, :]
    assert x_centered.shape == (n, m, d)

    x_centered_swp = np.moveaxis(x_centered, 0, 2)
    assert x_centered_swp.shape == (m, d, n)

    y_swp = np.linalg.solve(sigma, x_centered_swp)
    assert y_swp.shape == (m, d, n)

    y = np.moveaxis(y_swp, 2, 0)
    assert y.shape == (n, m, d)

    z = np.einsum('ijk,ijk->ij', x_centered, y)
    assert z.shape == (n, m)

    result = - 0.5 * dets[np.newaxis, :] - (d/2) * np.log(2*np.pi) -0.5 * z
    assert result.shape == (n, m)
    assert not np.isnan(result).any()

    return result

def E_step(x, log_pi, mu, sigma):
    n, d = x.shape
    k, _ = mu.shape
    assert mu.shape == (k, d)
    assert sigma.shape == (k, d, d)
    assert log_pi.shape == (k,)

    assert (log_pi < 0).all()

    log_tau_intermediate = log_gaussian_likelihood(x, mu, sigma)
    log_tau_intermediate += log_pi[np.newaxis, :]
    assert log_tau_intermediate.shape == (n, k)
    
    # Equivalent of :
    # tau = tau_intermediate / np.sum(tau_intermediate, axis=1)[:, np.newaxis]
    log_tau = log_tau_intermediate - logsumexp(log_tau_intermediate, axis=1, keepdims=True)
    return log_tau

def M_step(x, log_tau, reg=0):
    n, d = x.shape
    _, k = log_tau.shape
    assert log_tau.shape == (n, k)

    log_pi = -np.log(n) + scipy.special.logsumexp(log_tau, axis=0)
    assert log_pi.shape == (k,)
    
    normalized_tau = expnormalize(log_tau, axis=0)
    assert normalized_tau.shape == (n, k)
    assert (normalized_tau <= 1).all()
    assert (normalized_tau >= 0).all()
    assert np.isclose(np.sum(normalized_tau, axis=0), 1).all(), np.sum(normalized_tau, axis=0)

    mu = np.sum(normalized_tau[:,:,np.newaxis] * x[:,np.newaxis,:], axis=0)
    assert mu.shape == (k, d)

    x_centered = x[:, np.newaxis, :] - mu[np.newaxis, :, :]
    assert x_centered.shape == (n, k, d)

    sigma = np.einsum('ijk,ijl->jkl', normalized_tau[:,:,np.newaxis] * x_centered, x_centered)
    assert sigma.shape == (k, d, d)

    indices = np.arange(d)
    sigma[:, indices, indices] += reg
    assert (np.trace(sigma, axis1=1, axis2=2) >= d*reg).all()
    signs, dets = np.linalg.slogdet(sigma)
    assert (signs == 1).all(), (signs, dets, reg)

    return log_pi, mu, sigma

def batch_log_likelihood(x, log_pi, mu, sigma):
    n, d = x.shape
    k, = log_pi.shape
    assert mu.shape == (k, d)
    assert sigma.shape == (k, d, d)

    log_likelihoods = log_gaussian_likelihood(x, mu, sigma)
    assert log_likelihoods.shape == (n, k)

    log_likelihood_sample = logsumexp(log_pi[np.newaxis, :] + log_likelihoods, axis=1)
    assert log_likelihood_sample.shape == (n,)
    
    return log_likelihood_sample

def sample_log_likelihood(x, log_pi, mu, sigma):
    return np.sum(batch_log_likelihood(x, log_pi, mu, sigma))

def rel_err(new_log_pi, new_mu, new_sigma, log_pi, mu, sigma):
    err = np.linalg.norm(new_log_pi - log_pi) + np.linalg.norm(new_mu - mu) + np.linalg.norm(new_sigma - sigma)
    return err/(np.linalg.norm(log_pi) + np.linalg.norm(mu) + np.linalg.norm(sigma))

def kmeans_partition(x, centers):
    n, d = x.shape
    k, _ = centers.shape
    assert centers.shape == (k, d)

    dists = scipy.spatial.distance.cdist(x, centers)
    assert dists.shape == (n, k)

    center_indices = np.argmin(dists, axis=1)
    assert center_indices.shape == (n,)

    tau = np.zeros((n, k))

    indices = np.stack((np.arange(n), center_indices), axis=0)
    assert indices.shape == (2, n)
    
    tau[indices[0], indices[1]] = 1
    return tau

def kmeans_centers(x, k, tau):
    n, d = x.shape
    assert tau.shape == (n, k)

    centers = np.dot(tau.T, x)
    centers /= np.sum(tau, axis=0, keepdims=True).T
    assert centers.shape == (k, d)
    
    return centers
    
def kmeans(x, k, rtol=0, atol=0):
    _, d = x.shape

    centers = gen.choice(x, size=k, replace=False, axis=0)
    assert centers.shape == (k, d)
    
    stop = False
    while not stop:
        tau = kmeans_partition(x, centers)
        new_centers = kmeans_centers(x, k, tau)
        stop = np.allclose(centers, new_centers, rtol=rtol, atol=atol)
        centers = new_centers
    return tau

def map_cluster_membership(x, theta):
    n, _ = x.shape
    log_tau = E_step(x, *theta)
    cluster_membership = np.argmax(log_tau, axis=1)
    assert cluster_membership.shape == (n,)
    return cluster_membership

def EM_algorithm(x, init_theta, tol=1e-5, reg_sigma=0):
    # Init
    log_likelihoods = []
    theta = init_theta
    stop = False
    for it in tqdm(range(5000)):
        log_tau = E_step(x, *theta)
        new_theta = M_step(x, log_tau, reg=reg_sigma)

        stop = it > 10 and (rel_err(*new_theta, *theta) < tol)
        if stop:
            break
        theta = new_theta

        log_likelihoods.append(sample_log_likelihood(x, *theta))
    return theta, log_likelihoods

def GMM(x, k, tol=1e-5, reg_sigma=0, reg_init_tau=1e-15):
    init_tau = kmeans(x, k)
    # regularize
    init_tau = (init_tau + reg_init_tau) / (1 + k * reg_init_tau)
    assert (init_tau > 0).all()
    assert np.isclose(np.sum(init_tau, axis=1), 1).all()

    init_theta = M_step(x, np.log(init_tau), reg=reg_sigma)

    theta_est, log_likelihoods = EM_algorithm(x, init_theta, tol=tol, reg_sigma=reg_sigma)
    
    cluster_membership = map_cluster_membership(x, theta_est)

    return theta_est, cluster_membership, log_likelihoods

def spectral_clustering(affinity_matrix, k, n_eigenvectors, renormalize=None, **kwargs):
	n, _ = affinity_matrix.shape
	assert affinity_matrix.shape == (n, n)
	assert n_eigenvectors <= n

	_, eigenvectors = scipy.linalg.eigh(affinity_matrix, subset_by_index=(n-n_eigenvectors, n-1))
	assert eigenvectors.shape == (n, n_eigenvectors)

	if renormalize is not None:
		eigenvectors = renormalize(eigenvectors)
		assert eigenvectors.shape == (n, n_eigenvectors)

	_, cluster_membership, _ = GMM(eigenvectors, k, **kwargs)
	assert cluster_membership.shape == (n,)

	return cluster_membership
