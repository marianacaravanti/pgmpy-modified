import numpy as np 
import pandas as pd 
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.estimators import ExhaustiveSearch, K2Score
from pgmpy.estimators import HillClimbSearch, BicScore, BdeuScore
from pgmpy.estimators import HillClimbSearch
from pgmpy.estimators import BayesianEstimator
import csv
import scipy

def readData():
    #csv file containing temporal gene expression data
    data = pd.read_csv("/home/mariana/pgmpy/prog.csv", sep=",", header=0)
    return data

def printOutput(best_model)
	with open("network.csv", "wb") as f:
		writer = csv.writer(f)
		writer.writerows(best_model.edges())

def main():
	data = readData()
	labels = np.array(dataset.columns)
	datasetNp = np.array(dataset)

	data = pd.DataFrame(datasetNp, columns=labels)
	n = labels.shape[0]

	output = np.chararray(3, itemsize=10)

	model = BayesianModel()
	print "\nBayesian Network Inference with Temporal Data", 
    print "\nNetwork relations:  "

	hc = HillClimbSearch(data, scoring_method=BdeuScore(data))
	best_model = hc.estimate(tabu_length=10, max_indegree=3)
	print(best_model.edges())

	best_model.fit(data, BayesianEstimator)
	for cpd in best_model.get_cpds():
    	print(cpd)

    printOutput(best_model)

if __name__ == '__main__': # chamada da funcao principal
    main()
