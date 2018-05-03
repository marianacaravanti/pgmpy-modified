import csv
import numpy as np
import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.estimators import BdeuScore, BicScore
from pgmpy.estimators import HillClimbSearch
from pgmpy.estimators import BayesianEstimator
from pgmpy.estimators import DynamicBayesianEstimator
from pgmpy.models import DynamicBayesianNetwork as DBN

def readData():
    #csv file containing temporal gene expression data
    data = pd.read_csv("/home/mariana/pgmpy/prog.csv", sep=",", header=0)
    #csv file containing known relations between genes
    string = pd.read_csv("/home/mariana/pgmpy/bk-string.csv", sep=",", header=None)
    return data, string

def getData(data, labels):
    #function that returns the training data of the b0 network and the training data of the transition network 
    bList = []
    tList = []
    for x in range(0, data.shape[0]):
        row = (data.values[x])
        if row[0] == 0.0:
            bList.append(row)
    for x in range(0, data.shape[0]):
        row = (data.values[x])
        if not (np.isnan(row[0])) and row[0] != 0.0:
            tList.append(row)
    bData = pd.DataFrame(bList, columns=labels)
    bData.drop(bData.columns[[0]], axis=1, inplace=True)
    tData = pd.DataFrame(tList, columns=labels)
    tData.drop(tData.columns[[0]], axis=1, inplace=True)
    return bData, tData

def printOutputB(model):
    saida_bayesian = []
    for (X, Y) in model.edges():
        saida_bayesian.append((X,Y))
    with open("B_0_network.csv", "wb") as f:
        writer = csv.writer(f)
        writer.writerows(saida_bayesian)
   
def printOutputT(model):
    saida_transition = []
    for ((X,A),(Y,B)) in model.edges():
        saida_transition.append((X,Y))
    with open("B_transition.csv", "wb") as f:
        writer = csv.writer(f)
        writer.writerows(saida_transition)
    print "\nB_transition with cycles: "
    print saida_transition

def main():
    data, string = readData()
    genes = np.array(data.columns[1:])
    labels = np.array(data.columns)

    bayesianModel = BayesianModel()
    transitionModel = DBN()

    bayesianModel.add_nodes_from(genes)
    transitionModel.add_nodes_from(genes)

    bData, tData = getData(data, labels)
    
    print "\nDynamic Bayesian Network inference", 
    print "\nB_0 network relations:  "
    
    hcb = HillClimbSearch(bData, genes, scoring_method=BicScore(bData, labels, bk1=string, weight=4))
    best_model_b = hcb.estimate(start=bayesianModel, tabu_length=15, max_indegree=2)
    print(best_model_b.edges())

    printOutputB(best_model_b)

    print "\nLocal Probability Model: "
    best_model_b.fit(bData, BayesianEstimator)
    for cpd in best_model_b.get_cpds():
        print(cpd)

    print "\nB_transition network relations: "

    hct = HillClimbSearch(tData, genes, scoring_method=BicScore(tData, labels, bk1=string, weight=4))
    best_model_t = hct.estimate_dynamic(start=transitionModel, tabu_length=15, max_indegree=2)
    print(best_model_t.edges())

    printOutputT(best_model_t)

    print "\nLocal Probability Model: "
    best_model_t.fit(tData, BayesianEstimator)
    for cpd in best_model_t.get_cpds():
        print(cpd)
   
    
if __name__ == '__main__': # chamada da funcao principal
    main()
