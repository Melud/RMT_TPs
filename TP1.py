from tqdm import tqdm
from math import floor, sqrt
import numpy as np
import numpy.linalg as la
# from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
import tikzplotlib


def compute_R_N(N, prop=[1 / 3, 1 / 3, 1 / 3]):
	assert round(sum(prop), 6) == 1
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
# 	c = 1e-2
# 	N = 100
	c=0.25
	N=200
	n = floor(N / c)
	R_N = compute_R_N(N, prop=[.5, .5, .0])
	X_N = compute_X_N(N, n, distribution=0)
	print("création finie")
	res_matrix = 1 / n * X_N.T @ R_N @ X_N
	print("calcul de la matrice res fini")
	eig_vals = np.linalg.eigvalsh(res_matrix)
	print("calcul des vp finies")
	# observer uniquement les VP non nulles
	eig_vals = np.round(eig_vals, 6)
	eig_vals = eig_vals[eig_vals != 0]

	# plt.scatter(y=np.linspace(0, 0, len(eig_vals)), x=eig_vals)
	plt.hist(eig_vals,
			 bins=50,
			 # density=True
			 )
	plt.title(f"spectre de $X_N^T R_N X_N$ (excluant $0$) pour {n=}, {N=}")
	plt.show()


# tikzplotlib.save('exo1_plots.tex')


def exo2():
	c = 1e-1
	N = 100
	n = floor(N / c)
	c_n = N / n
	y = 1e-3

	prop = [.8, .1, .1]  # [1 / 3, 1 / 3, 1 / 3]  # [.5, .3, .2]#[1/3,1/3,1/3]
	assert sum(prop) == 1

	# R_N = compute_R_N(N, )
	# X_N = compute_X_N(N, n)
	print("création finie")
	# true_eig_vals = (
	# 		[1] * floor(prop[0] * N) +
	# 		[4] * floor(prop[1] * N) +
	# 		[7] * (N - floor(prop[0] * N) - floor(prop[1] * N))
	# )
	unique_true_eig_vals = np.array([1, 4, 7])

	# res_matrix = 1 / n * X_N.T @ R_N @ X_N
	# eig_vals = np.linalg.eigvalsh(res_matrix)

	# observer uniquement les VP non nulles
	## OU PAS ??????
	# eig_vals = np.round(eig_vals, 6)
	# eig_vals = eig_vals[eig_vals != 0]

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

	def function_x(t, c):
		if t == 0 or t in -1 / unique_true_eig_vals:
			x = None
		else:
			x = -1 / t + c * np.sum(
				prop * (unique_true_eig_vals / (1 + unique_true_eig_vals * t))
			)
		# x = -1 / t + c / len(eig_vals) * np.sum(eig_vals / (1 + eig_vals * t))
		return x

	# plot F
	nb_points = 500
	abscisses = np.linspace(0.1, 10, nb_points)
	values_of_F = [F(x) for x in abscisses]
	plt.plot(abscisses, values_of_F, ".-")
	plt.title("fonction $f$")
	# plt.show()
	tikzplotlib.save('exo2_plot_F.tex')

	# plot x
	# plt.figure(figsize=(16, 8))

	def plot_one_x(prop, c):

		nb_points = 100
		ε = 5e-2
		abscisses = np.linspace(-1.5, -1 - ε, nb_points)  # np.array([])
		for k in range(3):
			abscisses = np.concatenate([
				abscisses,
				np.linspace(-1 / unique_true_eig_vals[k] - ε, -1 / unique_true_eig_vals[k] + ε, nb_points)
			])
			if k < 2:
				abscisses = np.concatenate([
					abscisses,
					np.linspace(-1 / unique_true_eig_vals[k] + ε, -1 / unique_true_eig_vals[k + 1] - ε, nb_points)
				])
			else:  ##k=3
				abscisses = np.concatenate([
					abscisses,
					np.linspace(-1 / unique_true_eig_vals[k] + ε, 0 - ε, nb_points)
				])

		values_of_t_and_x = np.vstack(
			(abscisses,
			 np.array([function_x(t, c) for t in abscisses])
			 )
		)
		assert len(values_of_t_and_x.shape) == 2
		assert values_of_t_and_x.shape[0] == 2  # (2, )

		values_of_t_and_x = values_of_t_and_x[:, (values_of_t_and_x[1, :] != None)]
		assert len(values_of_t_and_x.shape) == 2
		# print(f"{values_of_t_and_x.shape=}")
		# values_of_t_and_x =
		plt.ylim((-1, 13))
		plt.plot(values_of_t_and_x[0],
				 values_of_t_and_x[1],
				 ".-",
				 ms=3,
				 label="$x(t)$")
		plt.title(f"fonction $x$ pour ${c=}, "
				  f"c_i=["
				  f"{prop[0]:.2f}, {prop[1]:.2f}, {prop[2]:.2f}"
				  f"]$",  # , {prop[1]:.2f}, {prop[2]:.2f}
				  fontsize=8)
		# fonction $x$ pour ${c=}, c_i=[{prop[0]:.2f}, {prop[1]:.2f}, {prop[2]:.2f}]$
		for eig_val in unique_true_eig_vals:
			plt.axvline(-1 / eig_val,
						color="red",
						# label=f"valeur propre {eig_val}"
						)
		# trouver où la fonction est croissante
		dérivée_discrète = np.diff(values_of_t_and_x[1, (values_of_t_and_x[0, :] < unique_true_eig_vals[0])])
		values_of_t_and_x = values_of_t_and_x[:, 1:]

		ind_croissance = dérivée_discrète > 0  # values_of_t_and_x[0, dérivée_discrète > 0]

		small_eps = 1e-2
		ind_croiss_dom_déf = ind_croissance
		for eig_val in unique_true_eig_vals:
			ind_croiss_dom_déf = np.logical_and(
				ind_croiss_dom_déf,
				np.logical_or(
					values_of_t_and_x[0, :] > -1 / eig_val + small_eps,
					values_of_t_and_x[0, :] < -1 / eig_val - small_eps,
				))
		plt.scatter(values_of_t_and_x[0, ind_croiss_dom_déf], [0] * np.sum(ind_croiss_dom_déf),
					color="purple",
					s=5,
					label="valeurs de $t$ pour lesquelles $x$ est croissante",
					zorder=-2
					)
		plt.axvline(0,
					color="lime",
					zorder=-1,
					label=f"Supp($F$)"
					)
		plt.scatter([0] * np.sum(ind_croiss_dom_déf),
					values_of_t_and_x[1, ind_croiss_dom_déf],
					s=5,
					color="white",
					# label="valeurs de $x$ pour lesquelles $x$ est croissante"
					)

		plt.legend(prop={'size': 6})

	def plot_x(i, j, prop, c):

		nb_points = 100
		ε = 5e-2
		abscisses = np.linspace(-1.5, -1 - ε, nb_points)  # np.array([])
		for k in range(3):
			abscisses = np.concatenate([
				abscisses,
				np.linspace(-1 / unique_true_eig_vals[k] - ε, -1 / unique_true_eig_vals[k] + ε, nb_points)
			])
			if k < 2:
				abscisses = np.concatenate([
					abscisses,
					np.linspace(-1 / unique_true_eig_vals[k] + ε, -1 / unique_true_eig_vals[k + 1] - ε, nb_points)
				])
			else:  ##k=2
				abscisses = np.concatenate([
					abscisses,
					np.linspace(-1 / unique_true_eig_vals[k] + ε, 0 - ε, 5 * nb_points)
				])

		values_of_t_and_x = np.vstack(
			(abscisses,
			 np.array([function_x(t, c) for t in abscisses])
			 )
		)
		assert len(values_of_t_and_x.shape) == 2
		assert values_of_t_and_x.shape[0] == 2  # (2, )

		values_of_t_and_x = values_of_t_and_x[:, (values_of_t_and_x[1, :] != None)]
		assert len(values_of_t_and_x.shape) == 2
		# print(f"{values_of_t_and_x.shape=}")
		# values_of_t_and_x =
		axes[i, j].set_ylim((-1, 13))
		axes[i, j].plot(values_of_t_and_x[0],
						values_of_t_and_x[1],
						".-",
						ms=3,
						label="$x(t)$")
		axes[i, j].set_title(f"fonction $x$ pour ${c=}, "
							 f"c_i=["
							 f"{prop[0]:.2f}, {prop[1]:.2f}, {prop[2]:.2f}"
							 f"]$",  # , {prop[1]:.2f}, {prop[2]:.2f}
							 fontsize=8)
		# fonction $x$ pour ${c=}, c_i=[{prop[0]:.2f}, {prop[1]:.2f}, {prop[2]:.2f}]$
		for eig_val in unique_true_eig_vals:
			axes[i, j].axvline(-1 / eig_val,
							   color="red",
							   # label=f"valeur propre {eig_val}"
							   )
		# trouver où la fonction est croissante
		dérivée_discrète = np.diff(values_of_t_and_x[1, (values_of_t_and_x[0, :] < unique_true_eig_vals[0])])
		values_of_t_and_x = values_of_t_and_x[:, 1:]

		ind_croissance = dérivée_discrète > 0  # values_of_t_and_x[0, dérivée_discrète > 0]

		small_eps = 1e-2
		ind_croiss_dom_déf = ind_croissance
		for eig_val in unique_true_eig_vals:
			ind_croiss_dom_déf = np.logical_and(
				ind_croiss_dom_déf,
				np.logical_or(
					values_of_t_and_x[0, :] > -1 / eig_val + small_eps,
					values_of_t_and_x[0, :] < -1 / eig_val - small_eps,
				))
		axes[i, j].scatter(values_of_t_and_x[0, ind_croiss_dom_déf], [0] * np.sum(ind_croiss_dom_déf),
						   color="purple",
						   s=5,
						   label="valeurs de $t$ pour lesquelles $x$ est croissante",
						   zorder=-2
						   )
		axes[i, j].axvline(0,
						   color="lime",
						   zorder=-1,
						   label=f"Supp($F$)"
						   )
		axes[i, j].scatter([0] * np.sum(ind_croiss_dom_déf),
						   values_of_t_and_x[1, ind_croiss_dom_déf],
						   s=6,
						   color="white",
						   # label="valeurs de $x$ pour lesquelles $x$ est croissante"
						   )

		axes[i, j].legend(prop={'size': 6})

	plt.figure()
	plot_one_x(prop, c)
	# plt.show()
	tikzplotlib.save(f'exo2_plots_x.tex')  # _{c=}_{prop=}
	plt.figure()
	ls_c = [1e-2, 1e-1]
	ls_prop = [[1 / 3, 1 / 3, 1 / 3], [.8, .1, .1]]
	# prop =  # [1 / 3, 1 / 3, 1 / 3]  # [.5, .3, .2]#[1/3,1/3,1/3]
	fig, axes = plt.subplots(len(ls_prop), len(ls_c))

	for i, prop in enumerate(ls_prop):
		for j, c in enumerate(ls_c):
			plot_x(i, j, prop, c)
	# plt.show()
	tikzplotlib.save(f'exo2_plots_4_x.tex')  # _{c=}_{prop=}
	return


def exo3():
	def iteration(N=100, i=0, naive=False):
		c = 1e-1
		n = floor(N / c)
		prop = [1 / 3, 1 / 3, 1 / 3]
		N1 = 0
		N2 = floor(N * 1 / 3)
		N3 = floor(N * 2 / 3)
		R_N = compute_R_N(N, prop)
		X_N = compute_X_N(N, n)
		Σ_n_star_Σ_n = 1 / n * X_N.T @ R_N @ X_N
		eig_vals_Σ_star_Σ = np.linalg.eigvalsh(Σ_n_star_Σ_n)

		#  besoin d’ enlèver les 0 ?  la matrice est de taille n×n et N<n
		eig_vals_Σ_star_Σ = np.round(eig_vals_Σ_star_Σ, 6)

		a = np.sqrt(1 / n * eig_vals_Σ_star_Σ)
		matrix_g = np.diag(eig_vals_Σ_star_Σ) - np.outer(a, a)
		eigvals_matrix_g = np.linalg.eigvalsh(matrix_g)

		eigvals_matrix_g = np.round(eigvals_matrix_g, 6)

		assert len(eig_vals_Σ_star_Σ) == n
		assert len(eigvals_matrix_g) == n

		assert np.all(eigvals_matrix_g <= eig_vals_Σ_star_Σ)
		diff_eig_vals = eig_vals_Σ_star_Σ - (eigvals_matrix_g if not naive else 0)

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
	if i == 0:
		eig_val = 1
	elif i == 1:
		eig_val = 4
	else:
		eig_val = 7
	ls_EQM = []
	ls_N = np.array([int(1.5 ** i) for i in np.linspace(10, 15, num=10)])
	for N in ls_N:
		ls_res_estimateurs = []
		for _ in tqdm(range(n_iters)):
			ls_res_estimateurs.append(iteration(N=N, i=i, naive=False))
		# print(ls_res_estimateurs)
		if N == ls_N[0]:
			plt.hist(ls_res_estimateurs, bins=100)
			plt.title(f"estimateurs de la valeur propre {eig_val} ({n_iters} exécutions)",
					  )
			plt.axvline(eig_val, color="red")
			tikzplotlib.save('TP1_ex3_hist.tex')
		erreur_quadratique = (np.array(ls_res_estimateurs) - eig_val) ** 2
		ls_EQM.append(np.mean(erreur_quadratique))
	# print(f"EQM = {np.mean(erreur_quadratique)}")
	plt.plot(ls_N,
			 ls_EQM,
			 ".-")
	plt.title(f"Erreur quadratique moyenne ({n_iters} itérations)")
	plt.xlabel(r"$N$",
			   fontsize=13
			   )
	plt.ylabel(r"$\mathbb{E}(|\hat\lambda_{i, N}^R-\lambda_{i}^R|^2)$",
			   fontsize=13
			   )
	#plt.show()
	tikzplotlib.save('TP1_ex3_EQM.tex')


def exo4():
	def iteration(n, N):
		prop = [1 / 2, 1 / 2, 0]
		N1 = 0
		N2 = floor(N * 1 / 2)
		R_N = compute_R_N(N, prop)
		X_N = compute_X_N(N, n)
		Σ_n_star_Σ_n = 1 / n * X_N.T @ R_N @ X_N
		eig_vals_Σ_star_Σ = np.linalg.eigvalsh(Σ_n_star_Σ_n)

		eig_vals_Σ_star_Σ = np.round(eig_vals_Σ_star_Σ, 6)

		a = np.sqrt(1 / n * eig_vals_Σ_star_Σ)
		matrix_g = np.diag(eig_vals_Σ_star_Σ) - np.outer(a, a)
		eigvals_matrix_g = np.linalg.eigvalsh(matrix_g)

		eigvals_matrix_g = np.round(eigvals_matrix_g, 6)

		assert len(eig_vals_Σ_star_Σ) == n
		assert len(eigvals_matrix_g) == n

		assert np.all(eigvals_matrix_g <= eig_vals_Σ_star_Σ)
		diff_eig_vals = eig_vals_Σ_star_Σ - eigvals_matrix_g

		def g_prime(array_λ, x):
			return 1/n*np.sum(1 / (array_λ - x) ** 2)

		A = n / N * np.sum(diff_eig_vals)
		B = n / N * np.sum([1 / g_prime(eig_vals_Σ_star_Σ[eig_vals_Σ_star_Σ != 0], x)
							for x in eigvals_matrix_g[eigvals_matrix_g != 0]])
		# A = 1
		# B = 200
		# print(f"{A=}")
		# print(f"{B=}")
		sqrt_Δ = np.sqrt(prop[0] / prop[1] * (B - A ** 2))#0 * 1j +
		# print(f"{sqrt_Δ=}")
		check_est_2 = A + sqrt_Δ
		check_est_1 = A - prop[1] / prop[0] * sqrt_Δ
		#print(check_est_1, check_est_2)

		hat_est_1 = n / (N * prop[0]) * np.sum(diff_eig_vals[n - N :n - N + N2])
		hat_est_2 = n / (N * prop[1]) * np.sum(diff_eig_vals[n - N + N2:])
		return check_est_1, check_est_2, hat_est_1, hat_est_2

	N = 200
	ls_c = np.linspace(0.3, 1, num=15)
	n_iters = 10
	ls_EQM = []
	eigvals = np.array([1,4,1,4])
	for c in ls_c:
		n = floor(N / c)
		ls_res = np.empty((4, n_iters))
		for it in tqdm(range(n_iters)):
			res = iteration(N=N, n=n)
			ls_res[:,it] = res
# 		print(ls_res)
# 		print(np.mean(ls_res, axis=-1))
# 		print(eigvals)
		erreur_quadratique = (np.array(ls_res) - eigvals[:, None]) ** 2
		ls_EQM.append(np.mean(erreur_quadratique, axis=-1))
		assert ls_EQM[-1].shape == (4,)
	eqm = np.array(ls_EQM)
	fig, ax = plt.subplots(1, 1)
	ax.plot(ls_c, eqm[:,2], label=r"$\mathbb{E}(|\hat\lambda_{1, N}^R-\lambda_{1}^R|^2)$")
	color = ax.lines[-1].get_color()
	ax.plot(ls_c, eqm[:,0], label=r"$\mathbb{E}(|\check\lambda_{1, N}^R-\lambda_{1}^R|^2)$", color=color, linestyle='dashed')

	ax.plot(ls_c, eqm[:,3], label=r"$\mathbb{E}(|\hat\lambda_{2, N}^R-\lambda_{2}^R|^2)$")
	color = ax.lines[-1].get_color()
	ax.plot(ls_c, eqm[:,1], label=r"$\mathbb{E}(|\check\lambda_{2, N}^R-\lambda_{2}^R|^2)$", color=color, linestyle='dashed')
	ax.set_xlabel("$c$")
	plt.title(f"Erreur quadratique moyenne ({n_iters} itérations)")
	ax.legend()
	#plt.show()
	tikzplotlib.save('TP1_ex4_EQM_check.tex')
	return


def main():
	exo3()
	return


if __name__ == '__main__':
	main()
