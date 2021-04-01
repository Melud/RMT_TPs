from tqdm import tqdm
from math import floor
import numpy as np
import numpy.linalg as la
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt


def compute_R_N(N, prop=[1 / 3, 1 / 3]):
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
	R_N = compute_R_N(N, prop=[.2, .7])
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

	R_N = compute_R_N(N, )
	X_N = compute_X_N(N, n)
	print("création finie")
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
			t = 1 / len(eig_vals) * np.sum(np.imag(1 / (-z * (1 + eig_vals * c_n * t) + (1 - c_n) * eig_vals)))
		return t

	def function_x(t):
		x = -1 / t + c / len(eig_vals) * np.sum(eig_vals / (1 + eig_vals * t))
		return x

	# plot F
	abscisses = np.linspace(0.1, 20, 1000)
	values_of_F = [F(x) for x in abscisses]
	plt.plot(abscisses, values_of_F, ".-")
	plt.title("fonction $f$")
	plt.show()
	# plot x
	abscisses = np.linspace(-1, 0, 1000)
	values_of_x = [function_x(t) for t in abscisses]
	plt.plot(abscisses, values_of_x, ".-")
	plt.title("fonction $x$")
	plt.show()

	return


def main():
	exo2()
	return


if __name__ == '__main__':
	main()
