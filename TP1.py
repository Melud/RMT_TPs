from tqdm import tqdm
from math import floor
import numpy as np
import numpy.linalg as la
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt


def compute_R_N(N, prop=[1 / 3, 1 / 3, 1 / 3]):
	assert sum(prop) == 1
	return np.diag([1] * floor(prop[0] * N) +
				   [4] * floor(prop[1] * N) +
				   [7] * (N - floor(prop[0] * N) - floor(prop[1] * N))
				   )


def compute_X_N(N, n, distribution=0):
	if distribution == 0:  # moment d’ordre 4 <∞
		return np.random.randn(N, n)
	elif distribution == 1:
		# moment d’ordre 4 = ∞
		# CENTRÉE RÉDUITE !
		return 1 / np.sqrt(3) * np.random.standard_t(df=3, size=(N, n))


def exo1():
	c = 1e-2
	N = 100
	n = floor(N / c)
	R_N = compute_R_N(N, prop=[.2, .7, .1])
	X_N = compute_X_N(N, n, distribution=1)
	print("création finie")
	res_matrix = 1 / n * X_N.T @ R_N @ X_N
	print("calcul de la matrice res fini")
	eig_vals = np.linalg.eigvalsh(res_matrix)
	print("calcul des vp finies")
	# observer uniquement les VP non nulles
	eig_vals = np.round(eig_vals, 6)
	eig_vals = eig_vals[eig_vals != 0]

	# plt.scatter(y=np.linspace(0, 0, len(eig_vals)), x=eig_vals)
	plt.hist(eig_vals, bins=50, density=True)
	plt.show()


def exo2():
	c = 1e-2
	N = 100
	n = floor(N / c)
	c_n = N / n
	y = 1e-3

	prop = [1 / 3, 1 / 3, 1 / 3]
	assert sum(prop) == 1

	R_N = compute_R_N(N, )
	X_N = compute_X_N(N, n)
	print("création finie")
	true_eig_vals = (
			[1] * floor(prop[0] * N) +
			[4] * floor(prop[1] * N) +
			[7] * (N - floor(prop[0] * N) - floor(prop[1] * N))
	)
	unique_true_eig_vals = np.array([1, 4, 7])
	res_matrix = 1 / n * X_N.T @ R_N @ X_N
	eig_vals = np.linalg.eigvalsh(res_matrix)

	# observer uniquement les VP non nulles
	## OU PAS ??????
	eig_vals = np.round(eig_vals, 6)
	eig_vals = eig_vals[eig_vals != 0]

	# print(f"N={N}, n={n}, number of eig_vals = {len(eig_vals)}")

	def F(x, n_iters=100):
		z = x + y * 1j
		t = 1
		for _ in range(n_iters):
			t = np.sum(
				prop * (
					np.imag(1 / (-z * (1 + unique_true_eig_vals * c_n * t) + (1 - c_n) * unique_true_eig_vals))
				)
			)
		# t = 1 / len(eig_vals) * np.sum(np.imag(1 / (-z * (1 + eig_vals * c_n * t) + (1 - c_n) * eig_vals)))
		return t

	def function_x(t):
		if t == 0 or t in -1 / unique_true_eig_vals:
			x = None
		else:
			x = -1 / t + c * np.sum(
				prop * (unique_true_eig_vals / (1 + unique_true_eig_vals * t))
			)
		# x = -1 / t + c / len(eig_vals) * np.sum(eig_vals / (1 + eig_vals * t))
		return x

	# plot F
	# nb_points = 500
	# abscisses = np.linspace(0.1, 10, nb_points)
	# values_of_F = [F(x) for x in abscisses]
	# plt.plot(abscisses, values_of_F, ".-")
	# plt.title("fonction $f$")
	# plt.show()
	# plot x
	nb_points = 100
	ε = 5e-2
	abscisses = np.array([])
	for i in range(3):
		abscisses = np.concatenate([
			abscisses,
			np.linspace(-1 / unique_true_eig_vals[i] - ε, -1 / unique_true_eig_vals[i] + ε, nb_points)
		])
		if i < 2:
			abscisses = np.concatenate([
				abscisses,
				np.linspace(-1 / unique_true_eig_vals[i] + ε, -1 / unique_true_eig_vals[i + 1] - ε, nb_points)
			])

	values_of_t_and_x = np.vstack(
		(abscisses,
		 np.array([function_x(t) for t in abscisses])
		 )
	)
	assert len(values_of_t_and_x.shape) == 2
	assert values_of_t_and_x.shape == (2, 500)

	values_of_t_and_x = values_of_t_and_x[:, (values_of_t_and_x[1, :] != None)]
	assert len(values_of_t_and_x.shape) == 2
	print(f"{values_of_t_and_x.shape=}")
	# values_of_t_and_x =
	plt.plot(values_of_t_and_x[0],
			 values_of_t_and_x[1],
			 ".-",
			 ms=3,
			 label="fonction $x$")
	plt.title("fonction $x$ et considérations")
	for eig_val in unique_true_eig_vals:
		plt.axvline(-1 / eig_val,
					color="red",
					# label=f"valeur propre {eig_val}"
					)
	# trouver où la fonction est croissante
	dérivée_discrète = np.diff(values_of_t_and_x[1, (values_of_t_and_x[0, :] < unique_true_eig_vals[0])])
	values_of_t_and_x = values_of_t_and_x[:, 1:]
	print(f"{(dérivée_discrète>0).shape=}")
	ind_croissance = dérivée_discrète > 0  # values_of_t_and_x[0, dérivée_discrète > 0]
	# C = abscisses_dérivée_discrète_croissante  # + 1 / unique_true_eig_vals.reshape(-1, 1)
	# ind = np.repeat(True, len(ind_dérivée_discrète_croissante))
	small_eps = 1e-2
	ind_croiss_dom_déf = ind_croissance
	for eig_val in unique_true_eig_vals:
		ind_croiss_dom_déf = np.logical_and(
			ind_croiss_dom_déf,
			np.logical_or(
				values_of_t_and_x[0, :] > -1 / eig_val + small_eps,
				values_of_t_and_x[0, :] < -1 / eig_val - small_eps,
			))
	# ind = np.where(
	# 	np.logical_or(C < small_eps, C > small_eps)
	# )[0]
	# ind_dérivée_discrète_croissante = ind_dérivée_discrète_croissante[ind]

	plt.scatter(values_of_t_and_x[0, ind_croiss_dom_déf], [0] * np.sum(ind_croiss_dom_déf),
				color="purple",
				label="valeurs de $t$ pour lesquelles $x$ est croissante")
	plt.scatter([0] * np.sum(ind_croiss_dom_déf),
				values_of_t_and_x[1, ind_croiss_dom_déf],
				color="navy",
				label="valeurs de $x$ pour lesquelles $x$ est croissante")
	plt.legend()
	# print(f"{ind_dérivée_discrète_croissante=}")
	# bounds =
	# for beg, end in zip(np.concatenate(([-np.inf], unique_true_eig_vals)),
	# 					np.concatenate((unique_true_eig_vals, [np.inf])),
	# 					):
	# 	abscisses_dérivée_discrète_croissante = \
	# 		np.concatenate((abscisses_dérivée_discrète_croissante,
	# 						np.diff(values_of_t_and_x[1,
	# 												  values_of_t_and_x[0, :] > beg and
	# 												  values_of_t_and_x[0, :] < end]))
	# 					   )
	# dérivée_discrète = np.diff(values_of_t_and_x[1, values_of_t_and_x[0, :] < unique_true_eig_vals[-1]])
	# print()
	plt.show()
	return


def exo3():
	def iteration(N=100, i=0):
		# N = 100
		c = 1e-1
		n = floor(N / c)
		# print(f"{n=},{N=}")
		prop = [1 / 3, 1 / 3, 1 / 3]
		N1 = 0
		N2 = floor(N * 1 / 3)
		N3 = floor(N * 2 / 3)
		R_N = compute_R_N(N, prop)
		# np.diag([1] * floor(prop[0] * N) +
		# 		[4] * floor(prop[1] * N) +
		# 		[7] * (N - floor(prop[0] * N) - floor(prop[1] * N))
		# 		)
		# sqrt_R = np.sqrt(R_N)
		X_N = compute_X_N(N, n)
		Σ_n_star_Σ_n = 1 / n * X_N.T @ R_N @ X_N
		eig_vals_Σ_star_Σ = np.linalg.eigvalsh(Σ_n_star_Σ_n)
		#  besoin d’ enlèver les 0 ?  la matrice est de taille n×n et N<n
		eig_vals_Σ_star_Σ = np.round(eig_vals_Σ_star_Σ, 6)
		# eig_vals_Σ_star_Σ = eig_vals_Σ_star_Σ[eig_vals_Σ_star_Σ != 0]
		# print(f"{len(eig_vals_Σ_star_Σ)=}")
		a = np.sqrt(1 / n * eig_vals_Σ_star_Σ)
		matrix_g = np.diag(eig_vals_Σ_star_Σ) - np.outer(a, a)
		eigvals_matrix_g = np.linalg.eigvalsh(matrix_g)

		eigvals_matrix_g = np.round(eigvals_matrix_g, 6)
		# on enlève les 0
		# eigvals_matrix_g = eigvals_matrix_g[eigvals_matrix_g != 0]

		assert len(eig_vals_Σ_star_Σ) == n
		assert len(eigvals_matrix_g) == n
		# print(f"{eigvals_matrix_g=}")
		# print(f"{eig_vals_Σ_star_Σ=}")
		assert (eigvals_matrix_g <= eig_vals_Σ_star_Σ).all()
		diff_eig_vals = eig_vals_Σ_star_Σ - eigvals_matrix_g

		# print(f"{diff_eig_vals}")
		def estimateur(i):
			if i == 0:
				return n / (N * prop[0]) * np.sum(diff_eig_vals[n - N + N1:n - N + N2])
			elif i == 1:
				return n / (N * prop[1]) * np.sum(diff_eig_vals[n - N + N2:n - N + N3])
			else:
				return n / (N * prop[2]) * np.sum(diff_eig_vals[n - N + N3:])

		return estimateur(i=i)

	n_iters = 10
	i = 1
	ls_EQM = []
	ls_N = np.array([2 ** i for i in range(6, 9)])
	for N in ls_N:
		ls_res_estimateurs = []
		for _ in tqdm(range(n_iters)):
			ls_res_estimateurs.append(iteration(N=N, i=i))
		# print(ls_res_estimateurs)
		# plt.hist(ls_res_estimateurs)
		# plt.axvline(1, color="red")
		# plt.show()
		erreur_quadratique = (np.array(ls_res_estimateurs) - 1) ** 2
		ls_EQM.append(np.mean(erreur_quadratique))
	# print(f"EQM = {np.mean(erreur_quadratique)}")
	plt.plot(ls_N,
			 ls_EQM,
			 ".-")
	plt.show()


# print(f"{n / (N * prop[0]) * np.sum(diff_eig_vals[n-N+N1:n-N+N2])=}")
# print(f"{n / (N * prop[1]) * np.sum(diff_eig_vals[n - N + N2:n - N + N3])=}")
# print(f"{n / (N * prop[2]) * np.sum(diff_eig_vals[n - N + N3:])=}")


def main():
	exo3()
	return


if __name__ == '__main__':
	main()
