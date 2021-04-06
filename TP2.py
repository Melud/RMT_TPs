import matplotlib.pyplot as plt
import tikzplotlib
import numpy as np
import numpy.testing as npt
import scipy
import scipy.linalg
import math
import itertools

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
	ls_q0 = [0.3, 0.4, 0.5]
	ls_eps = [i * .1 * 0.4 for i in range(1, 4)]
	ls_q1_q2 = [(.4, .6), (.2, .6), (0.1, 0.6)]
	ls_ls_q = [[(f"q_0={q0}", q0 * np.ones(n)) for q0 in ls_q0], \
			[(f"\epsilon={eps:.2f}", (0.4 - eps) + 2 * eps * np.random.rand(n)) for eps in ls_eps], \
		   [(f"(q_1, q_2) = ({q1}, {q2})", np.random.choice([q1, q2], n)) for q1, q2 in ls_q1_q2]]
	diag_M =20
	ls_M = [(f"(\eta_1, \eta_2) = ({eta1}, {eta2})", np.array([[diag_M, eta2, eta1], [eta2, diag_M+1, eta1], [eta1, eta1, diag_M+2]]))
			for eta1, eta2 in [(.1, .9), (.3, .6), (.0, .0)]]
	cardinaux_classes = [n // 3, n // 3, n - 2 * (n // 3)]
	for ls_index, ls_q in enumerate(ls_ls_q):
		fig, axes = plt.subplots(len(ls_q), len(ls_M))
		for i, (descr_q, q) in enumerate(ls_q):
			for j, (descr_M, M) in enumerate(ls_M):
				_, _, eig_vals, eig_vects, J = model(q, M, cardinaux_classes)
				axes[i, j].hist(eig_vals, bins=n // 10)  # , density=True)
				title = "$"+ descr_M + ", " + descr_q + "$"
				axes[i, j].set_title(title)
				print(title)
				print(f"{np.max(eig_vals)=}")
				print(eig_vects[-3:] @ J)
		#plt.show()
		tikzplotlib.save(f"TP2_Obs_{ls_index}.tex")
	return

def compute_isolated_eigvals(q0, M, prop_classes):
	diag_M = np.diagonal(M)
	sigma = math.sqrt(q0**2*(1-q0**2))
	rho = q0**2/sigma * diag_M * prop_classes
	return sigma*(rho + 1/rho)[rho > 1], rho

def cas_homogene():
	n = 1000
	ls_q0 = [0.3, 0.4, 0.5]
	ls_q = [q0 * np.ones(n) for q0 in ls_q0]
	K = 3
	ls_diag_M = [15, 17.5, 20]
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
			repr_M = np.array2string(np.diagonal(M), separator=', ')[1:-1]
			axes[i, j].set_title(f"$q_0={q[0]:.2f}, M=diag({repr_M})$")
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
	tikzplotlib.save("TP2_Cas_Homogene_EigVal.tex")
	return

def cas_homogene_growing_n():
	ls_n = list(range(600, 2200, 200))
	q0 = 0.4
	K = 3
	diag_M = 20
	M = np.diagflat([diag_M, diag_M+1, diag_M+2])
	prop_classes = [1/4, 1/4, 1/2]
	empirical_eigvals = np.empty((len(ls_n), K))
	empirical_alignments = np.empty_like(empirical_eigvals)
	for index_n, n in enumerate(ls_n):
		q = q0 * np.ones(n)
		cardinaux_classes = [int(round(n * prop)) for prop in prop_classes[:-1]]
		cardinaux_classes.append(n - sum(cardinaux_classes))
		cardinaux_classes = np.array(cardinaux_classes)
		memberships, aff_matrix, eig_vals, _, J = model(q, M, cardinaux_classes, eigvects=False)
		empirical_eigvals[index_n] = eig_vals[-3:]
		_, isolated_eigvect = scipy.linalg.eigh(aff_matrix, subset_by_index=(n-1,n-1))
		normalized_J = J/np.sqrt(cardinaux_classes[None,:])
		observed = np.abs(isolated_eigvect[:,0] @ normalized_J)
		empirical_alignments[index_n] = observed
	fig, axes = plt.subplots(1, 2)
	isolated_eigvals, rho = compute_isolated_eigvals(q[0], M, prop_classes)
	sigma = math.sqrt(q[0]**2*(1-q[0]**2))
	assert len(isolated_eigvals) == K
	for i in range(K):
		axes[0].plot(ls_n, empirical_eigvals[:, i], label=f"Observed $\lambda_{i+1}$")
		color = axes[0].lines[-1].get_color()
		axes[0].axhline(isolated_eigvals[i], label=f"Predicted $\lambda_{i+1}$", color=color, linestyle='dashed')
		axes[1].plot(ls_n, empirical_alignments[:, i], label=f"Observed $|j_{i+1}^* v_{1}|/\sqrt{{ n_{i+1} }}$")
		color = axes[1].lines[-1].get_color()
		axes[1].axhline(np.sqrt(1 - 1/rho[-1]**2) if i == K-1 else 0,
				label=f"Predicted $|j_{i+1}^* v_{1}|/\sqrt{{ n_{i+1} }}$", color=color, linestyle='dashed')
	axes[0].legend()
	axes[1].legend()
	axes[0].set_xlabel('n')
	axes[1].set_xlabel('n')
	#plt.show()
	tikzplotlib.save('TP2_Cas_homogene_growing_n.tex')
	return

def precision(k, memberships, inferred_memberships):
	ls_precisions = []
	for cluster_indices in itertools.permutations(range(k)):
		correct = [
				np.sum(inferred_memberships[memberships == i] == cluster_indices[i])
				for i in range(k)
				]
		ls_precisions.append(sum(correct)/len(memberships))
	return max(ls_precisions)

def community_detection_homogeneous():
	n = 2000
	ls_q0 = [0.3, 0.4, 0.5]
	ls_q = [q0 * np.ones(n) for q0 in ls_q0]
	K = 3
	repeat = 5
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
			res = 0
			for _ in range(repeat):
				memberships, aff_matrix, eig_vals, _, J = model(q, M, cardinaux_classes, eigvects=False)
				inferred_memberships = spectral_clustering(aff_matrix, K, K, tol=1e-7)
				p = precision(K, memberships, inferred_memberships)
				res += p/repeat
			print(f"Precision = {res*100}%")
	return

def display_spectrum(matrix):
	eigvals = np.linalg.eigvalsh(matrix)
	plt.figure()
	plt.hist(eigvals, bins=len(eigvals)//10)
	plt.show()

def community_detection_heterogeneous():
	n = 2000
	q0 = 0.4
	ls_q1_q2 = [(0.1, 0.8)]
	ls_q = \
		[np.random.choice([q1, q2], n) for q1, q2 in ls_q1_q2]
	K = 3
	repeat = 5
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
			res = [0, 0, 0, 0]
			for _ in range(repeat):
				memberships, aff_matrix, eig_vals, _, J = model(q, M, cardinaux_classes, eigvects=False)

				inferred_memberships = spectral_clustering(aff_matrix, K, K, tol=1e-7)
				p = precision(K, memberships, inferred_memberships)
				res[0] += p/repeat

				renormalized_aff_matrix = aff_matrix / np.outer(q, q)
				inferred_memberships = spectral_clustering(renormalized_aff_matrix, K, K, tol=1e-7, reg_sigma=1e-16)
				p = precision(K, memberships, inferred_memberships)
				res[1] += p/repeat

				renormalized_aff_matrix = aff_matrix / np.sqrt(np.outer(q, q))
				inferred_memberships = spectral_clustering(renormalized_aff_matrix, K, K, tol=1e-7, reg_sigma=1e-16)
				p = precision(K, memberships, inferred_memberships)
				res[2] += p/repeat

				renormalize = lambda eigenvectors : 1/q[:, None] * eigenvectors
				inferred_memberships = spectral_clustering(aff_matrix,
										K, K, renormalize=renormalize, tol=1e-7, reg_sigma=1e-16)
				p = precision(K, memberships, inferred_memberships)
				res[3] += p/repeat
			print(res)
	return



def main():
#	observations_preliminaires()
#	cas_homogene()
#	cas_homogene_growing_n()
#	community_detection_homogeneous()
	community_detection_heterogeneous()
	return


if __name__ == '__main__':
	main()

# np.linalg.eigvalsh()
