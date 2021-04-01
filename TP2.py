import matplotlib.pyplot as plt
import numpy as np


def model(q, M, cardinaux_classes):
	# q is a column… or not
	n = len(q)
	appartenances = np.concatenate([[i] * cardinaux_classes[i] for i in range(len(cardinaux_classes))])
	np.random.shuffle(appartenances)
	# print(appartenances)
	C = 1 + 1 / np.sqrt(n) * M
	E = np.outer(q, q) * C[appartenances, :][:, appartenances]
	A = np.random.binomial(n=1, p=E)
	# np.fill_diagonal(A, 0)
	A = np.triu(A, k=1) + np.triu(A, k=1).T
	B = A - np.outer(q, q)
	val_p = np.linalg.eigvalsh(1 / np.sqrt(n) * B)
	return val_p


def observations_préliminaires():
	# K = 3
	n = 600
	q0 = 0.1
	ls_eps = [i * .1 * min(q0, (1 - q0)) for i in range(1, 4)]  # [.1, .2, .3]
	ls_q1_q2 = [(.4, .6), (.2, .8), (.3, .7)]
	ls_q = [q0 * np.ones(n)] + \
		   [(q0 - eps) + 2 * eps * np.random.rand(n) for eps in ls_eps] + \
		   [np.random.choice([q1, q2], n) for q1, q2 in ls_q1_q2]
	ls_M = [np.array([[10, eta2, eta1], [eta2, 10, eta1], [eta1, eta1, 10]])
			for eta1, eta2 in [(.1, .9), (.3, .6), (.0, .0)]]
	cardinaux_classes = [n // 3, n // 3, n - 2 * (n // 3)]
	fig, axes = plt.subplots(len(ls_q), len(ls_M))
	for i, q in enumerate(ls_q):
		for j, M in enumerate(ls_M):
			eig_vals = model(q, M, cardinaux_classes)
			axes[i, j].hist(eig_vals, bins=n // 10)  # , density=True)
			axes[i, j].set_title(f"{i, j}")
			print(f"({i},{j})")
			print(f"{np.max(eig_vals)}")
	plt.show()
	# print(axes.shape)
	# axes[i,j].plot()
	return


def main():
	observations_préliminaires()
	return


if __name__ == '__main__':
	main()

# np.linalg.eigvalsh()
