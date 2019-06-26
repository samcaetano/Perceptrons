import numpy as np
import argparse
import math
import neural_models as models # this is for loading the perceptrons
from wrapper import Loader # this is for data manipulation
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help="Select a dataset to train the models", default="random")
parser.add_argument("--mode", help="Train or evaluate the model", default="train")
parser.add_argument("--model", help="Select a neural model (mlp, grnn, committee)", default="grnn")
args=parser.parse_args()

def build_model(model, num_samples, num_features, num_classes):
	lr = 0.001
	if args.model == "mlp":
		# Multi Layer Perceptron
		return models.MultiPerceptron(num_samples=num_samples,
			num_features=num_features, num_classes=num_classes), False

	elif args.model == "grnn":
		# Global Recurrent Network
		return models.GlobalRecurrentNet(num_samples=num_samples,
			num_features=num_features, num_classes=num_classes), False

	elif args.model == "mlp_committee":
		m1 = models.MultiPerceptron(num_samples=num_samples, h=1,
			num_features=num_features, num_classes=num_classes)
		m2 = models.MultiPerceptron(num_samples=num_samples, h=2,
			num_features=num_features, num_classes=num_classes)
		m3 = models.MultiPerceptron(num_samples=num_samples, h=4,
			num_features=num_features, num_classes=num_classes)
		m4 = models.MultiPerceptron(num_samples=num_samples, h=16,
			num_features=num_features, num_classes=num_classes)

		return models.Committee(experts=[m1, m2, m3, m4]), True


	elif args.model == "mlp+grnn_committee":
		m1 = models.MultiPerceptron(num_samples=num_samples, h=1,
			num_features=num_features, num_classes=num_classes)
		m2 = models.MultiPerceptron(num_samples=num_samples, h=2,
			num_features=num_features, num_classes=num_classes)
		m3 = models.GlobalRecurrentNet(num_samples=num_samples, h=1,
			num_features=num_features, num_classes=num_classes)
		m4 = models.GlobalRecurrentNet(num_samples=num_samples, h=2,
			num_features=num_features, num_classes=num_classes)

		return models.Committee(experts=[m1, m2, m3, m4]), True

def training(model, Xtr, Ytr, Xval, Yval):	
	EPOCHS = 100

	errors, val_errors = [], []

	prev_error = 0
	prev_model = None
	count = 3

	for i in range(EPOCHS):
		if count == 0: 
			model = prev_model
			break

		if i > 0: 
			prev_error = error
			prev_model = model

		pred_y, error = model.train(Xtr, Ytr)
		errors.append(error)

		if error == prev_error: count -= 1

		_, val_error = model.eval(Xval, Yval)
		val_errors.append(val_error)
		
		if val_error > error and prev_model != None: 
			model = prev_model
			break

		if i % 500 == 0:
			print("{}/{} epoch: loss {}, val loss {}".format(i+1, EPOCHS, error, val_error))
		elif i % 100 == 0:
			print("{}/{} epoch: loss {}".format(i+1, EPOCHS, error))

	# Plot results
	plt.plot(errors)
	plt.plot(val_errors)
	plt.legend(["Training", "Validation"])
	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	plt.savefig("figures/{}+{}+{}.png".format(args.dataset, args.model, model.h))
	plt.close()

	return pred_y

def evaluating(model, X, Y):
	print("Not implemented yet =(")

def main():
	# Load data
	loader = Loader()

	if args.dataset == "random":
		X, Y = loader.random_samples()
	else:
		# Load and normalize the data
		Xtr, Ytr, Xts, Yts, Xval, Yval = loader.load(path=args.dataset, lag=1)

	num_samples = Xtr.shape[0] # M
	num_features = Xtr.shape[-1] # N
	num_classes = Ytr.shape[-1] # k

	print("# samples >> {}".format(num_samples))
	print("# features >> {}".format(num_features))
	print("# lookahead >> {}".format(num_classes))


	model, is_committee = build_model(args.model, num_classes=num_classes, 
		num_features=num_features, num_samples=num_samples)

	if args.mode == "train":
		if not is_committee:
			# Train the model
			_ = training(model, Xtr, Ytr, Xval, Yval)
			_, test_error = model.eval(Xts, Yts)
			print("Test loss {}".format(test_error))
		else:
			committee = model # rename

			# Train the committee
			experts_y = []
			for expert in committee.experts:
				pred_y = training(expert, Xtr, Ytr, Xval, Yval)
				experts_y.append(pred_y)

			P, L, variances = committee.combine_experts(X=Xtr, Y=Ytr, Xts=Xts, Yts=Yts, Ypred=experts_y)

			#variances *= -1

			plt.plot(L)
			plt.legend(["Expert 1", "Expert 2", "Expert 3", "Expert 4"])
			plt.ylabel("Likelihood")
			plt.xlabel("Step")
			plt.savefig("figures/likelihood_{}+{}.png".format(args.dataset, args.model))
			plt.close()

			plt.plot(variances)
			plt.legend(["Expert 1", "Expert 2", "Expert 3", "Expert 4"])
			plt.ylabel("Variance")
			plt.xlabel("Step")
			plt.savefig("figures/variance_{}+{}.png".format(args.dataset, args.model))
			plt.close()			



	elif args.mode == "evaluate":
		# Evaluate the model
		evaluating(model, Xtr, Ytr)
main()