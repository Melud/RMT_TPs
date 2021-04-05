import matplotlib.pyplot as plt
import tikzplotlib
import numpy as np
import numpy.testing as npt
import scipy
import scipy.linalg
import math

from em_algorithm import spectral_clustering
from test_em import most_frequent


def model(q, M, cardinaux_classes, eigvects=True):
	# q is a column… or not
	n = len(q)
	K = len(cardinaux_classes)
	appartenances = np.concatenate([[i] * cardinaux_classes[i] for i in range(len(cardinaux_classes))])
	np.random.shuffle(appartenances)
	# print(appartenances)
	C = 1 + 1 / np.sqrt(n) * M
	E = np.outer(q, q) * C[appartenances, :][:, appartenances]
	assert (E >= 0).all()
	assert (E <= 1).all()
	A = np.random.binomial(n=1, p=E)
	# np.fill_diagonal(A, 0)
	A = np.triu(A, k=1) + np.triu(A, k=1).T
	B = A - np.outer(q, q)
	if eigvects:
		val_p, vect_p = np.linalg.eigh(1 / np.sqrt(n) * B)
		vect_p = vect_p.T
	else:
		val_p, vect_p = np.linalg.eigvalsh(1 / np.sqrt(n) * B), None
	J = np.zeros((n, K))
	J[np.arange(n), appartenances] = 1
	npt.assert_allclose(J @ C @ J.T, C[appartenances, :][:, appartenances])
	return appartenances, 1 / np.sqrt(n) * B, val_p, vect_p, J


def observations_preliminaires():
	# K = 3
	n = 1000
	q0 = 0.05
	ls_eps = [i * .1 * min(q0, (1 - q0)) for i in range(1, 4)]  # [.1, .2, .3]
	ls_q1_q2 = [(.4, .6), (0.2, 0.8), (.3, .7)]
	ls_q = [q0 * np.ones(n)] + \
		   [(q0 - eps) + 2 * eps * np.random.rand(n) for eps in ls_eps] + \
		   [np.random.choice([q1, q2], n) for q1, q2 in ls_q1_q2]
	diag_M = 15
	ls_M = [np.array([[diag_M, eta2, eta1], [eta2, diag_M+1, eta1], [eta1, eta1, diag_M+2]])
			for eta1, eta2 in [(.1, .9), (.3, .6), (.0, .0)]]
	cardinaux_classes = [n // 3, n // 3, n - 2 * (n // 3)]
	fig, axes = plt.subplots(len(ls_q), len(ls_M))
	for i, q in enumerate(ls_q):
		for j, M in enumerate(ls_M):
			_, _, eig_vals, eig_vects, J = model(q, M, cardinaux_classes)
			axes[i, j].hist(eig_vals, bins=n // 10)  # , density=True)
			axes[i, j].set_title(f"{i, j}")
			print(f"({i},{j})")
			print(f"{np.max(eig_vals)=}")
			print(eig_vects[-3:] @ J)
	plt.show()
	# print(axes.shape)
	# axes[i,j].plot()
	return

def compute_isolated_eigvals(q0, M, prop_classes):
	diag_M = np.diagonal(M)
	sigma = math.sqrt(q0**2*(1-q0**2))
	rho = q0**2/sigma * diag_M * prop_classes
	return sigma*(rho + 1/rho)[rho > 1], rho

def cas_homogene():
	n = 1000
	ls_q0 = np.linspace(0.1, 0.3, num=4)
	ls_q = [q0 * np.ones(n) for q0 in ls_q0]
	K = 3
	ls_diag_M = [10, 12, 15]
	ls_M = [np.diagflat([diag_M, diag_M+1, diag_M+2]) for diag_M in ls_diag_M]
	prop_classes = [1/4, 1/4, 1/2]
	cardinaux_classes = [int(round(n * prop)) for prop in prop_classes[:-1]]
	cardinaux_classes.append(n - sum(cardinaux_classes))
	cardinaux_classes = np.array(cardinaux_classes)
	fig, axes = plt.subplots(len(ls_q), len(ls_M), constrained_layout=True)
	for i, q in enumerate(ls_q):
		for j, M in enumerate(ls_M):
			print(f"({i=},{j=})")
			print(f"$q_0$={q[0]} M = diag({np.diagonal(M)})")
			memberships, aff_matrix, eig_vals, _, J = model(q, M, cardinaux_classes, eigvects=False)
			isolated_eigvals, rho = compute_isolated_eigvals(q[0], M, prop_classes)
			print(f"{np.max(eig_vals)=}")
			print(f"{isolated_eigvals=}")
			axes[i, j].hist(eig_vals, bins=n // 10)  # , density=True)
			sigma = math.sqrt(q[0]**2*(1-q[0]**2))
			axes[i, j].set_title(f"$q_0$={q[0]:.2f} 2$\sigma$={2*sigma:.2f} M=diag({np.diagonal(M)})")
			for isolated_eigval in isolated_eigvals:
				axes[i, j].axvline(isolated_eigval, color='red', label=f"{isolated_eigval:.2f}")
			axes[i, j].axvline(2*sigma, color='green', label=f"2$\sigma$={2*sigma:.2f}")

			_, isolated_eigvect = scipy.linalg.eigh(aff_matrix, subset_by_index=(n-1,n-1))
			normalized_J = J/np.sqrt(cardinaux_classes[None,:])
			observed = np.abs(isolated_eigvect[:,0] @ normalized_J)
			predicted = np.diagflat(np.sqrt(1 - 1/rho**2))[-1]
			print(f"Observed =\n{observed}")
			print(f"Predicted =\n{predicted}")
	#plt.show()
	tikzplotlib.save('test.tex')
	return

def precision(k, memberships, inferred_memberships):
	inferred_cluster_indices = [
			most_frequent(inferred_memberships[memberships == i]) for i in range(k)
			]
	errors = [
			np.sum(inferred_memberships[memberships == i] == inferred_cluster_indices[i])
			for i in range(k)
			]
	return sum(errors)/len(memberships)

def community_detection_homogeneous():
	n = 2000
	ls_q0 = [0.3, 0.4, 0.5]
	ls_q = [q0 * np.ones(n) for q0 in ls_q0]
	K = 3
	ls_diag_M = [15, 17.5, 20]
	ls_M = [np.diagflat([diag_M, diag_M+1, diag_M+2]) for diag_M in ls_diag_M]
	prop_classes = [1/4, 1/4, 1/2]
	cardinaux_classes = [int(round(n * prop)) for prop in prop_classes[:-1]]
	cardinaux_classes.append(n - sum(cardinaux_classes))
	cardinaux_classes = np.array(cardinaux_classes)
	for i, q in enumerate(ls_q):
		for j, M in enumerate(ls_M):
			print(f"({i=},{j=})")
			print(f"$q_0$={q[0]} M = diag({np.diagonal(M)})")
			memberships, aff_matrix, eig_vals, _, J = model(q, M, cardinaux_classes, eigvects=False)

			inferred_memberships = spectral_clustering(aff_matrix, K, K, tol=1e-7)
			p = precision(K, memberships, inferred_memberships)
			print(f"Precision = {p*100}%")
	return

def community_detection_heterogeneous():
	n = 2000
	q0 = 0.4
	ls_q1_q2 = [(0.01, 0.8)]
	ls_q = \
		[np.random.choice([q1, q2], n) for q1, q2 in ls_q1_q2]
	K = 3
	ls_diag_M = [20]
	ls_M = [np.diagflat([diag_M, diag_M+1, diag_M+2]) for diag_M in ls_diag_M]
	prop_classes = [1/4, 1/4, 1/2]
	cardinaux_classes = [int(round(n * prop)) for prop in prop_classes[:-1]]
	cardinaux_classes.append(n - sum(cardinaux_classes))
	cardinaux_classes = np.array(cardinaux_classes)
	for i, q in enumerate(ls_q):
		for j, M in enumerate(ls_M):
			print(f"({i=},{j=})")
			print(f"{q=} M = diag({np.diagonal(M)})")
			memberships, aff_matrix, eig_vals, _, J = model(q, M, cardinaux_classes, eigvects=False)

			inferred_memberships = spectral_clustering(aff_matrix, K, K, tol=1e-7)
			p = precision(K, memberships, inferred_memberships)
			print(f"Precision = {p*100}%")

			inferred_memberships = spectral_clustering(1/q[:, None] * aff_matrix * 1/q[None, :], K, K, tol=1e-7, reg_sigma=1e-16)
			p = precision(K, memberships, inferred_memberships)
			print(f"Improved method 1: Precision = {p*100}%")

			renormalize = lambda eigenvectors : np.sqrt(1/q[:, None]) * eigenvectors
			inferred_memberships = spectral_clustering(1/q[:, None] * aff_matrix * 1/q[None, :],
									K, K, renormalize=renormalize, tol=1e-7, reg_sigma=1e-16)
			p = precision(K, memberships, inferred_memberships)
			print(f"Improved method 2: Precision = {p*100}%")
	return



def main():
	#observations_preliminaires()
	#cas_homogene()
	#community_detection_homogeneous()
	community_detection_heterogeneous()
	return


if __name__ == '__main__':
	main()

# np.linalg.eigvalsh()
