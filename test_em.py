import pytest
import numpy as np
gen = np.random.default_rng()
import em_algorithm as em

n = 150
d = 5
k = 3

def test_log_gaussian_likelihood():
    x = np.zeros((n, d))
    mu = np.zeros((k, d))
    sigma = np.array([np.eye(d)] * 3)
    assert np.allclose(em.log_gaussian_likelihood(x, mu, sigma), np.log(1/(np.sqrt((2 * np.pi) ** d)) * np.ones((n,k))))

@pytest.fixture
def x():
    x1 = gen.normal(loc=-2, size=(n//k, d))
    x2 = gen.normal(loc= 0, size=(n//k, d))
    x3 = gen.normal(loc= 2, size=(n//k, d))
    return np.concatenate((x1, x2, x3), axis=0)

def most_frequent(a):
    return np.argmax(np.bincount(a))

def check_clusters(cluster_membership):
    clusters = [most_frequent(cluster_membership[i * n//k : (i+1) * n//k]) for i in range(k)]
    assert sorted(clusters) == list(range(k))
    errors = [np.sum(cluster_membership[i * n//k : (i+1)*n//k] != clusters[i]) for i in range(k)]
    return sum(errors)/n

def test_em(x):
    init_tau = em.kmeans(x, k)
    cluster_membership = np.argmax(init_tau, axis=1)
    err_kmeans = check_clusters(cluster_membership)
    # regularize
    reg_init_tau = 1e-20
    reg_sigma = 0
    init_tau = (init_tau + reg_init_tau) / (1 + k * reg_init_tau)
    assert (init_tau > 0).all()
    assert np.isclose(np.sum(init_tau, axis=1), 1).all()

    init_theta = em.M_step(x, np.log(init_tau), reg=reg_sigma)

    theta, log_likelihoods = em.EM_algorithm(x, init_theta, reg_sigma=reg_sigma, tol=1e-7)

    for old, new in zip(log_likelihoods[:-1], log_likelihoods[1:]):
        assert old <= new + 1e-20
    cluster_membership = em.map_cluster_membership(x, theta)
    err_gmm = check_clusters(cluster_membership)
    assert err_gmm    <= 5/100
    assert err_kmeans <= 5/100


