import numpy as np

class SinglePerceptron:
   def __init__(self, num_features, num_classes, num_samples, learning_rate=0.001):
      self.M = num_samples
      self.N = num_features
      self.k = num_classes

      self.bias = np.random.randint(1,2, size=(1, self.k))
      self.bias[:] = 1

      self.lr = learning_rate

      self.W = np.random.uniform(-1, 1, (self.N, self.k))
      self.W *= .5

   def activation(self, function, Z, deriv=False):
      Z = np.array(Z, dtype=np.float128)

      if function=="sigmoid":
         if deriv: return Z * (1 - Z)
         return 1. / (1. + np.exp(-Z))

      elif function=="tanh":
         if deriv: return 1 - self.activation(Z=Z, function="tanh")**2
         return (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))

      elif function=="linear":
         return Z

   def train(self, Xtrain, Ytrain):
      # Feedforward prop
      _y= np.sum(np.dot(Xtrain, self.W)+self.bias, axis=1) # (M,)
      y = self.activation(Z=_y, function="sigmoid") # (M,)
      
      loss = Ytrain - y #
      
      squared_loss = loss**2 

      j = .5*np.sum(squared_loss, axis=1) 

      # Erro (quadratico medio) total
      J = self.M**(-1) * np.sum(j) # 

      # Backward prop
      djde = np.sum(loss, axis=1)   # (M,) 
      d_ydW = np.sum(Xtrain, axis=1) # (M,)
      dyd_y = self.activation(Z=_y, deriv=True, function="sigmoid") # (M,k)

      dJdW = (djde*d_ydW).dot(dyd_y) # (k,)

      self.W = self.W + self.lr*dJdW

      return y, J

class MultiPerceptron:
	def __init__(self, num_features, num_classes, num_samples, h=3, learning_rate=0.001):
		self.M = num_samples
		self.h = h
		self.N = num_features
		self.k = num_classes
		
		self.biasU = np.random.randint(1,2, size=(1, self.h))
		self.biasU[:] = 1

		self.biasV = np.random.randint(1,2, size=(1, self.k))
		self.biasV[:] = 1

		self.lr = learning_rate

		self.U = np.random.uniform(-1, 1, (self.N, self.h))
		self.U *= .5

		self.V = np.random.uniform(-1, 1, (self.h, self.k))
		self.V *= .5

	def activation(self, function, Z, deriv=False):
		Z = np.array(Z, dtype=np.float128)
		if function=="sigmoid":
			if deriv: return Z * (1 - Z)
			return 1. / (1. + np.exp(-Z))

		elif function=="tanh":
			if deriv: return 1 - self.activation(Z=Z, function="tanh")**2
			return (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))

		elif function=="linear":
			return Z

		elif function=="relu":
			return np.maximum(Z, 0)

	def train(self, X, Y):
		# Feedforward prop
		_z = np.dot(X, self.U)+self.biasU
		z = self.activation(Z=_z, function="tanh") # (M,h)
		
		_y = np.dot(z, self.V) + self.biasV
		y = self.activation(Z=_y, function="tanh") # ()
		
		loss = Y - y # (M,k)
		
		squared_loss = loss**2 # (k, )

		j = .5*np.sum(squared_loss, axis=1) # (M,)

		# Erro (quadratico medio) total
		J = self.M**(-1) * np.sum(j) # 

		# Backward prop
		djde = np.sum(loss, axis=1) 	# (M,)
		d_ydV = np.sum(z, axis=1) 		# (M,)
		d_zdU = np.sum(X, axis=1)	# (M,)
		d_ydz = np.sum(self.V, axis=0)	# (k,)
		
		dyd_y = self.activation(Z=_y, deriv=True, function="tanh") # (M,k)
		dzd_z = self.activation(Z=_z, deriv=True, function="tanh") # (M,h)

		dJdV = (djde*d_ydV).dot(dyd_y) # (k,)

		dJdU = (djde*d_zdU*(d_ydz.dot(dyd_y.T))).dot(dzd_z)

		self.V = self.V + self.lr*dJdV
		self.U = self.U + self.lr*dJdU

		return y, J#

	def eval(self, X, Y):
		# Feedforward prop
		_z = np.dot(X, self.U)+self.biasU
		z = self.activation(Z=_z, function="tanh") # (M,h)
		
		_y = np.dot(z, self.V) + self.biasV
		y = self.activation(Z=_y, function="tanh") # ()
		
		loss = Y - y # (M,k)
		
		squared_loss = loss**2 # (k, )

		j = .5*np.sum(squared_loss, axis=1) # (M,)

		# Erro (quadratico medio) total
		J = self.M**(-1) * np.sum(j) # 

		return y, J

class LocalRecurrentNet:
	def __init__(self, num_features, num_classes, num_samples, h=3, learning_rate=0.001):
		self.M = num_samples # this could be the num_samples
		self.h = h
		self.N = num_features
		self.k = num_classes
		self.lr = learning_rate
		
		self.U = np.random.uniform(-1, 1, (self.N, self.h)) * .5
		self.biasU = np.random.randint(1, 2, size=(1, self.h))
		self.biasU[:] = 1

		self.V = np.random.uniform(-1, 1, (self.h, self.k)) * .5
		self.biasV = np.random.randint(1,2, size=(1, self.k))
		self.biasV[:] = 1

		self.A = np.random.uniform(-1, 1, (self.h, self.h)) * .5
		self.A = np.diag(self.A) # diagonal matrix for local recurrency

		self.z_prev = np.zeros((self.h,)) # previous z (delay)
		self.dz_prev = np.ones((self.M,)) # previous partial derivative of z


	def activation(self, function, Z, deriv=False):
		Z = np.array(Z, dtype=np.float128)
		if function=="sigmoid":
			if deriv: return Z * (1 - Z)
			return 1. / (1. + np.exp(-Z))

		elif function=="tanh":
			if deriv: return 1 - self.activation(Z=Z, function="tanh")**2
			return (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))

		elif function=="linear":
			return Z

	def train(self, Xtrain, Ytrain):
		# Feedforward prop
		_z = np.dot(Xtrain, self.U) + np.dot(self.A, self.z_prev) + self.biasU # (M,h)
		
		z = self.activation(Z=_z, function="sigmoid") # (M,h)
		
		# Updates the previous z normalizing by the mean
		self.z_prev = self.M**-1 * np.sum(z, axis=0) # (h,)

		_y = np.dot(z, self.V) + self.biasV # (h, k)
		y = self.activation(Z=_y, function="tanh") # (h, k)
		
		loss = Ytrain - y # (M,k)
		
		squared_loss = loss**2 # (k, )

		j = .5*np.sum(squared_loss, axis=1) # (M,)

		# Erro (quadratico medio) total
		J = self.M**(-1) * np.sum(j) # (1)

		# Backward prop
		djde = np.sum(loss, axis=1) 	# (M,)
		dyd_y = self.activation(Z=_y, deriv=True, function="tanh") # (M,k)
		d_ydV = np.sum(z, axis=1) 		# (M,)

		d_ydz = np.sum(self.V, axis=0)	# (k,)
		dzd_z = self.activation(Z=_z, deriv=True, function="sigmoid") # (M,h)
		d_zdU = np.sum(Xtrain, axis=1) + self.dz_prev # (M,)
		
		d_zdA = self.dz_prev

		dJdV = (djde*d_ydV).dot(dyd_y) # (k,)
		dJdU = (djde*d_zdU*(d_ydz.dot(dyd_y.T))).dot(dzd_z)
		dJdA = (djde*d_zdA*(d_ydz.dot(dyd_y.T))).dot(dzd_z)

		self.dz_prev = d_zdA

		self.V = self.V + self.lr*dJdV
		self.U = self.U + self.lr*dJdU
		self.A = self.A + self.lr*dJdA

		return y, J#

class GlobalRecurrentNet:
	def __init__(self, num_features, num_classes, num_samples, h=3, learning_rate=0.001):
		self.M = num_samples 
		self.h = h
		self.N = num_features
		self.k = num_classes
		self.lr = learning_rate
		
		self.U = np.random.uniform(-1, 1, (self.N, self.h)) * .5
		self.biasU = np.random.randint(1, 2, size=(1, self.h))
		self.biasU[:] = 1

		self.V = np.random.uniform(-1, 1, (self.h, self.k)) * .5
		self.biasV = np.random.randint(1,2, size=(1, self.k))
		self.biasV[:] = 1

		self.A = np.random.uniform(-1, 1, (self.h, self.h)) * .5
		self.biasA = np.random.randint(1,2, size=(1, self.k))
		self.biasA[:] = 1

		self.z_prev = np.zeros((self.h,)) # previous z (delay)
		self.dz_prev = np.ones((self.M,)) # previous partial derivative of z


	def activation(self, function, Z, deriv=False):
		Z = np.array(Z, dtype=np.float128)
		if function=="sigmoid":
			if deriv: return Z * (1 - Z)
			return 1. / (1. + np.exp(-Z))

		elif function=="tanh":
			if deriv: return 1 - self.activation(Z=Z, function="tanh")**2
			return (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))

		elif function=="linear":
			return Z

		elif function=="relu":
			return np.maximum(Z, 0)

	def train(self, X, Y):
		# Feedforward prop
		_z = np.dot(X, self.U) + (np.dot(self.A, self.z_prev) + self.biasA) + self.biasU # (M,h)
		
		z = self.activation(Z=_z, function="relu") # (M,h)
		
		# Updates the previous z normalizing by the mean
		self.z_prev = self.M**-1 * np.sum(z, axis=0) # (h,)

		_y = np.dot(z, self.V) + self.biasV # (h, k)
		y = self.activation(Z=_y, function="relu") # (h, k)
		
		loss = Y - y # (M,k)
		
		squared_loss = loss**2 # (k, )

		j = .5*np.sum(squared_loss, axis=1) # (M,)

		# Erro (quadratico medio) total
		J = self.M**(-1) * np.sum(j) # (1)

		# Backward prop
		djde = np.sum(loss, axis=1) 	# (M,)
		dyd_y = self.activation(Z=_y, deriv=True, function="relu") # (M,k)
		d_ydV = np.sum(z, axis=1) 		# (M,)

		d_ydz = np.sum(self.V, axis=0)	# (k,)
		dzd_z = self.activation(Z=_z, deriv=True, function="relu") # (M,h)
		d_zdU = np.sum(X, axis=1) + self.dz_prev # (M,)
		
		d_zdA = self.dz_prev

		dJdV = (djde*d_ydV).dot(dyd_y) # (k,)
		dJdU = (djde*d_zdU*(d_ydz.dot(dyd_y.T))).dot(dzd_z)
		dJdA = (djde*d_zdA*(d_ydz.dot(dyd_y.T))).dot(dzd_z)

		self.dz_prev = d_zdA

		self.V = self.V + self.lr*dJdV
		self.U = self.U + self.lr*dJdU
		self.A = self.A + self.lr*dJdA

		return y, J#

	def eval(self, X, Y):
		# Feedforward prop
		_z = np.dot(X, self.U) + np.dot(self.A, self.z_prev) + self.biasU # (M,h)
		
		z = self.activation(Z=_z, function="relu") # (M,h)
		
		# Updates the previous z normalizing by the mean
		self.z_prev = self.M**-1 * np.sum(z, axis=0) # (h,)

		_y = np.dot(z, self.V) + self.biasV # (h, k)
		y = self.activation(Z=_y, function="relu") # (h, k)
		
		loss = Y - y # (M,k)
		
		squared_loss = loss**2 # (k, )

		j = .5*np.sum(squared_loss, axis=1) # (M,)

		# Erro (quadratico medio) total
		J = self.M**(-1) * np.sum(j) # (1)

		return y, J

class Committee:
	def __init__(self, experts):
		self.experts = experts
		self.m = len(self.experts)
		self.N = self.experts[0].N
		self.W = np.random.uniform(-1, 1, (self.m, self.N)) * .5
		self.X = None
		self.Y = None
		self.Ypred = None
		self.var = np.ones((self.m,))
		self.Ls = []

	def __gating(self, X):
		self.X = X
		z = self.X.dot(self.W.T) # (M, N)x(N,m) -> (M,m)

		return np.exp(z) / np.sum(np.exp(z), axis=0) # (M, m) / (m,) -> (M,m)

	def combine_experts(self, X, Y, Ypred, Xts, Yts):
		self.Y = Y
		self.Ypred = np.array(Ypred) # (m, M, k)
		#print("Ypred", self.Ypred.shape)
		Yg = self.__gating(X) # (M,m)
		#print("Yg", Yg.shape)
		Ym = Yg * self.Ypred.T # (m,M)x(k,M,m) -> (k,M,m)
		#print("Ym", Ym.shape)

		# likelihood calc
		diff = self.Y - self.Ypred # (m, M, k)
		#print("diff", diff.shape)
		P = np.sum(np.exp((-diff * diff.T) / 2 * self.var), axis=0) # (M, m)
		L = np.sum(np.log(Yg * P), axis=0)
		print("Likelihoods. Expert 1: {}, Expert 2: {}, Expert 3: {}, Expert 4: {}".format(
			L[0], L[1], L[2], L[3]))
		self.Ls.append(L)

		prev_L, nit, nitmax = 0, 0, 10
		variances = []

		while nit < nitmax:
			nit += 1

			h_aux = Yg * P # (M,m)
			#print("h_aux", h_aux.shape)
			h = h_aux / np.sum(h_aux, axis=0) # (M,m)

			self.maximise_gating(h)

			for i, expert in enumerate(self.experts): self.var[i] = self.maximise_expert(expert, h, self.var[i])
			variances.append(self.var.copy())
			#print("variances", self.var)

			prev_L = L

			Yg = self.__gating(X) # (M,m)

			experts_y = []
			for expert in self.experts:
				y, _ = expert.train(self.X, self.Y)
				experts_y.append(y)

			Ym = np.sum(Yg.T.dot(self.Ypred), axis=1) # (m,M)x(M,k) -> (m,k)

			# likelihood calc
			diff = self.Y - self.Ypred # (m, M, k)
			P = np.sum(np.exp((-diff * diff.T) / 2 * self.var), axis=0) # (M, m)
			L = np.sum(np.log(Yg * P), axis=0)
			#L = np.log(np.sum(Yg * P, axis=1)) # ()

			print("Step {}/{}. Expert 1: {}, Expert 2: {}, Expert 3: {}, Expert 4: {}".format(
				nit, nitmax, L[0], L[1], L[2], L[3]))
			self.Ls.append(L)

		for i, expert in enumerate(self.experts):
			_, test_loss = expert.eval(Xts, Yts)
			print("Expert {}: test loss {}".format(i+1, test_loss))

		return P, self.Ls, np.array(variances)


	def maximise_gating(self, h):
		Yg = self.__gating(self.X) # (M,m)
		grad = (h - Yg).T.dot(self.X / self.X.shape[0]) #(M,m)-(M,m)x(M,N) -> (m,N)
		#print("maximise_gating grad",grad.shape)

		nit = 0
		grad_aux = grad
		while nit < 1000:
			nit += 1
			self.W = self.W + 0.1*grad_aux # (m,N)+(m,N)
			Yg = self.__gating(self.X) # (M,m)
			grad = (h - Yg).T.dot(self.X / self.X.shape[0])
			grad_aux = grad

	def maximise_expert(self, expert, h, var):
		Ye, _ = expert.train(self.X, self.Y) # (M, k)

		h /= var # (M,m)
		#print("maximise_expert h",h.shape)

		diff = self.Y - Ye # (M,k)
		#print("maximise_expert diff", diff.shape)

		grad = (diff * h).T.dot(self.X / self.X.shape[0])# ((M,k)x(M,m))'x(M,N) -> (m,N)
		#print("maximise_expert grad",grad.shape)

		nit = 0
		grad_aux = grad
		while nit < 1000:
			nit += 1
			expert.U = expert.U + 0.1*np.sum(grad_aux) # (N,h) + (m,N)
			Ye, _ = expert.train(self.X, self.Y)
			grad = (diff * h).T.dot(self.X / self.X.shape[0])
			grad_aux = grad

		diff = self.Y - Ye # (M, k)
		soma = np.sum(h.T.dot(-diff * diff), axis=1) # (M,m)x(M,k) -> (m,k) -> (m,)
		#print("maximise_expert soma", soma.shape)
		
		#var = max(0.05, np.sum(((1/self.Y.shape[-1])*soma) / np.sum(h)))
		var = np.sum(((1/self.Y.shape[-1])*soma) / np.sum(h))

		return var