import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Loader:
	def load(self, path, lag):
		if path == "serie1":
			df = pd.read_csv("datasets/serie1_trein.txt", sep="\n", header=None)

		elif path == "serie2":
			df = pd.read_csv("datasets/serie2_trein.txt", sep="\n", header=None)

		elif path == "serie3":
			df = pd.read_csv("datasets/serie3_trein.txt", sep="\n", header=None)

		elif path == "serie4":
			df = pd.read_csv("datasets/serie4_trein.txt", sep="\n", header=None)

		plt.plot(df)

		data = list(x for x in df[0])

		data = self.normalize(X=data, method="z-score")
		plt.plot(data)
		
		plt.legend(["Unnormalized", "Normalized"])
		plt.savefig("figures/{}.png".format(path))
		plt.close()

		X, Y = self.split_by_lag(data, lag)

		return self.split((X, Y))

	def random_samples(self):
		# To load random samples
		M, timesteps, k = 100, 5, 1

		X = np.random.rand(M, timesteps)
		Y = np.random.randint(0, 2, size=(M, k))
		X /= 255 # normalize
		return X, Y

	def normalize(self, X, method):
		X = np.array(X)
		if method == "z-score":
			z = (X - np.mean(X)) / np.std(X)

		elif method == "interval":
			z = (X - np.min(X)) / (np.max(X) - np.min(X))

		return z

	def split(self, data):
		X, Y = data[0], data[1]

		testing_size = int(0.25 * X.shape[0])

		Xtr = X[:-testing_size]
		Xts = X[-testing_size:]
		Ytr = Y[:-testing_size]
		Yts = Y[-testing_size:]

		validation_size = int(0.25 * Xtr.shape[0])


		Xval = Xtr[-validation_size:]
		Yval = Ytr[-validation_size:]

		Xtr = Xtr[:-validation_size]
		Ytr = Ytr[:-validation_size]

		return Xtr, Ytr, Xts, Yts, Xval, Yval

	def split_by_lag(self, data, L):
		# Split the data by the lag timestep given
		X, Y, i = [], [], 0
		while i < len(data)-L-1:
			start = i
			end = start+L
			X.append(data[start:end])
			Y.append([data[end+1]])
			i += 1
		# arrumar formatos de x e y
		return np.array(X), np.array(Y)