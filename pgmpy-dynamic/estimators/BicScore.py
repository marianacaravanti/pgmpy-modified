#!/usr/bin/env python

from math import log
import pandas as pd
import numpy as np

from pgmpy.estimators import StructureScore


class BicScore(StructureScore):
    def __init__(self, data, nodes=None, bk1=None, bk2=None, weight=0, **kwargs):
        """
        Class for Bayesian structure scoring for BayesianModels with Dirichlet priors.
        The BIC/MDL score ("Bayesian Information Criterion", also "Minimal Descriptive Length") is a
        log-likelihood score with an additional penalty for network complexity, to avoid overfitting.
        The `score`-method measures how well a model is able to describe the given data set.

        Parameters
        ----------
        data: pandas DataFrame object
            datafame object where each column represents one variable.
            (If some values in the data are missing the data cells should be set to `numpy.NaN`.
            Note that pandas converts each column containing `numpy.NaN`s to dtype `float`.)

        nodes: list of strings 
        	string list containing the data nodes
        
        bk1 : pandas DataFrame object (optional)
        	 biological knowledge - data frame object containing three columns. The first and second columns contain a node name and 
        	 the third column contain a score that represents the degree of the relationship between these two nodes.
		
		bk2 : pandas DataFrame object (optional)
        	 biological knowledge - data frame object containing three columns. The first and second columns contain a node name and 
        	 the third column contain a score that represents the degree of the relationship between these two nodes.
		
        weigth: integer number (default: 0)
        	integer representing the weight assigned to the biological knowledge equation. 
        
        state_names: dict (optional)
            A dict indicating, for each variable, the discrete set of states (or values)
            that the variable can take. If unspecified, the observed values in the data set
            are taken to be the only possible states.
        
        complete_samples_only: bool (optional, default `True`)
            Specifies how to deal with missing data, if present. If set to `True` all rows
            that contain `np.Nan` somewhere are ignored. If `False` then, for each variable,
            every row where neither the variable nor its parents are `np.NaN` is used.
            This sets the behavior of the `state_count`-method.

        References
        ---------
        [1] Koller & Friedman, Probabilistic Graphical Models - Principles and Techniques, 2009
        Section 18.3.4-18.3.6 (esp. page 802)
        [2] AM Carvalho, Scoring functions for learning Bayesian networks,
        http://www.lx.it.pt/~asmc/pub/talks/09-TA/ta_pres.pdf
        """
        if nodes is None:
            self.nodes = self.state_names.keys()
        else:
            self.nodes = nodes
        if bk1 is not None:
            self.bk1 = bk1
        else:
            self.bk1 = None
        if bk2 is not None:
            self.bk2 = bk2
        else:
            self.bk2 = None
        self.weight = weight
        super(BicScore, self).__init__(data, **kwargs)

    #--------------------------------------------------------------------------------------------------------------------
    """ Implementation of calc_t_data function for Dynamic Bayesian Networks 
        Contribution by Mariana C. Souza """
    #--------------------------------------------------------------------------------------------------------------------
    def calc_t_data(self, variable, parents):
    	"""
        Function that returns a dataFrame containing the target variable data and the parents variables data,
        where the target gene column undergoes a sift.
        """ 
        t_data = pd.DataFrame(self.data[variable])
        t_data.ix[:, variable] = t_data.ix[:, variable].shift(-1)

        for X,Y in parents:
            if Y is 0:
                t_data[X,Y] = self.data[X]
            else:
                t_data[X,Y] = self.data.ix[:, X].shift(-1)
        return t_data
   
	#--------------------------------------------------------------------------------------------------------------------
    """ Extending the local_score function adding biological knowlegde  
        Contribution by Mariana C. Souza """
    #--------------------------------------------------------------------------------------------------------------------
    def local_score(self, variable, parents, bData=None):
        "Computes a score that measures how much a \
        given variable is \"influenced\" by a given list of potential parents."
        if self.bk1 is not None:
            bk1 = self.bk1
        else:
            bk1 = None
        if self.bk2 is not None:
            bk2 = self.bk2
        else:
            bk2 = None
        weight = self.weight

        var_states = self.state_names[variable]
        var_cardinality = len(var_states)
        state_counts = self.state_counts(variable, parents, bData)
        sample_size = len(bData)
        num_parents_states = float(len(state_counts.columns))

        score = 0

        for parents_state in state_counts:  # iterate over df columns (only 1 if no parents)
            conditional_sample_size = sum(state_counts[parents_state])

            for state in var_states:
                if state_counts[parents_state][state] > 0:
                    score += state_counts[parents_state][state] * (log(state_counts[parents_state][state]) -
                                                                   log(conditional_sample_size))
        if bk1 is None and bk2 is None:
            score -= 0.5 * log(sample_size) * num_parents_states * (var_cardinality - 1)
        else:
            soma_score = 0
            num_parents = 0
            """ 
            Verifies if there is a relation between the variable and its parents in the first vector of biological 
            knowledge and adds the score to the calculation.
            """
            if bk1 is not None:
                for i in range(0, bk1.shape[0]):
                    (X, Y, pontuacao) = bk1.values[i]
                    for parent in parents:
                        if (X,Y) == (parent, variable):
                            soma_score = soma_score + pontuacao
                            num_parents += 1
            """ 
            Verifies if there is a relation between the variable and its parents in the second vector of biological 
            knowledge and adds the score to the calculation.            
            """            
            if bk2 is not None:
                for i in range(0, bk2.shape[0]):
                    (X, Y, pontuacao) = bk2.values[i]
                    for parent in parents:
                        if (X,Y) == (parent, variable):
                            soma_score = soma_score + pontuacao
                            num_parents += 1


            if soma_score != 0:
                soma_score = soma_score / num_parents

            score -= 0.5 * log(sample_size) * num_parents_states * (var_cardinality - 1) - (weight * soma_score)
        return score

    #--------------------------------------------------------------------------------------------------------------------
    """ Implementation of local_score function with biological knowlegde for Dynamic Bayesian Networks 
        Contribution by Mariana C. Souza """
    #--------------------------------------------------------------------------------------------------------------------
    def local_score_dynamic(self, variable, parents):
    	if self.bk1 is not None:
            bk1 = self.bk1
        else:
            bk1 = None
        if self.bk2 is not None:
            bk2 = self.bk2
        else:
            bk2 = None
        weight = self.weight
        
        "Computes a score that measures how much a \
        given variable is \"influenced\" by a given list of potential parents."

        tData = self.calc_t_data(variable, parents)

        var_states = self.state_names[variable]
        var_cardinality = len(var_states)
        state_counts = self.state_counts_dynamic(variable, parents, tData)

        sample_size = 0.0
        for parents_state in state_counts:  # iterate over df columns (only 1 if no parents)
            for state in var_states:
                sample_size += state_counts[parents_state][state]

        num_parents_states = float(len(state_counts.columns))

        score = 0

        for parents_state in state_counts:  # iterate over df columns (only 1 if no parents)
            conditional_sample_size = sum(state_counts[parents_state])

            for state in var_states:
                if state_counts[parents_state][state] > 0:
                    score += state_counts[parents_state][state] * (log(state_counts[parents_state][state]) -
                                                                   log(conditional_sample_size))

        if bk1 is None and bk2 is None:
            score -= 0.5 * log(sample_size) * num_parents_states * (var_cardinality - 1)
        else:
            """ 
            Verifies if there is a relation between the variable and its parents in the first vector of biological 
            knowledge and adds the score to the calculation.
            """
            soma_score = 0.0
            num_parents = 0
            if bk1 is not None:
                for i in range(0, bk1.shape[0]):
                    (X, Y, pontuacao) = bk1.values[i]

                    for (parent,t) in parents:

                        if (X, Y) == (parent, variable):
                            soma_score = soma_score + pontuacao
                            num_parents += 1
            """ 
            Verifies if there is a relation between the variable and its parents in the second vector of biological 
            knowledge and adds the score to the calculation.
            """
            if bk2 is not None:
                for i in range(0, bk2.shape[0]):
                    (X, Y, pontuacao) = bk2.values[i]

                    for (parent, t) in parents:
                        if (X, Y) == (parent, variable):
                            soma_score = soma_score + pontuacao
                            num_parents += 1


            if soma_score != 0:
                soma_score = soma_score / num_parents

            score -= 0.5 * log(sample_size) * num_parents_states * (var_cardinality - 1) - (weight*soma_score)

        return score
