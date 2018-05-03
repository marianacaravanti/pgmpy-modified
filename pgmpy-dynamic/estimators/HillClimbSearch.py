#!/usr/bin/env python
from itertools import permutations
from itertools import product
import networkx as nx

from pgmpy.estimators import StructureEstimator, K2Score
from pgmpy.models import BayesianModel
from pgmpy.models import DynamicBayesianNetwork as DB


class HillClimbSearch(StructureEstimator):
    def __init__(self, data, nodes=None, scoring_method=None, **kwargs):
        """
        Class for heuristic hill climb searches for BayesianModels, to learn
        network structure from data. `estimate` attempts to find a model with optimal score.

        Parameters
        ----------
        data: pandas DataFrame object
            datafame object where each column represents one variable.
            (If some values in the data are missing the data cells should be set to `numpy.NaN`.
            Note that pandas converts each column containing `numpy.NaN`s to dtype `float`.)

        nodes: List of strings containing the data node names.
        scoring_method: Instance of a `StructureScore`-subclass (`K2Score` is used as default)
            An instance of `K2Score`, `BdeuScore`, or `BicScore`.
            This score is optimized during structure estimation by the `estimate`-method.

        state_names: dict (optional)
            A dict indicating, for each variable, the discrete set of states (or values)
            that the variable can take. If unspecified, the observed values in the data set
            are taken to be the only possible states.

        complete_samples_only: bool (optional, default `True`)
            Specifies how to deal with missing data, if present. If set to `True` all rows
            that contain `np.Nan` somewhere are ignored. If `False` then, for each variable,
            every row where neither the variable nor its parents are `np.NaN` is used.
            This sets the behavior of the `state_count`-method.
        """
        if scoring_method is not None:
            self.scoring_method = scoring_method
        else:
            self.scoring_method = K2Score(data, **kwargs)
        if nodes is None:
            self.nodes = self.state_names.keys()
        else:
            self.nodes = nodes

        super(HillClimbSearch, self).__init__(data, **kwargs)

    def _legal_operations(self, model, tabu_list=[], max_indegree=None):
        """Generates a list of legal (= not in tabu_list) graph modifications
        for a given model, together with their score changes. Possible graph modifications:
        (1) add, (2) remove, or (3) flip a single edge. For details on scoring
        see Koller & Fridman, Probabilistic Graphical Models, Section 18.4.3.3 (page 818).
        If a number `max_indegree` is provided, only modifications that keep the number
        of parents for each node below `max_indegree` are considered."""

        local_score = self.scoring_method.local_score
        #calcula_b_data = self.scoring_method.calcula_b_data
        nodes = self.nodes
        potential_new_edges = (set(permutations(nodes, 2)) -
                               set(model.edges()) -
                               set([(Y, X) for (X, Y) in model.edges()]))

        for (X, Y) in potential_new_edges:  # (1) add single edge
            if nx.is_directed_acyclic_graph(nx.DiGraph(model.edges() + [(X, Y)])):
                operation = ('+', (X, Y))
                if operation not in tabu_list:
                    old_parents = model.get_parents(Y)
                    new_parents = old_parents + [X]
                    if max_indegree is None or len(new_parents) <= max_indegree:
                        #bData = calcula_b_data(Y)
                        score_delta = local_score(Y, new_parents, self.data) - local_score(Y, old_parents, self.data)
                        yield(operation, score_delta)

        for (X, Y) in model.edges():  # (2) remove single edge
            operation = ('-', (X, Y))
            if operation not in tabu_list:
                old_parents = model.get_parents(Y)
                new_parents = old_parents[:]
                new_parents.remove(X)
                #bData = calcula_b_data(Y)
                score_delta = local_score(Y, new_parents, self.data) - local_score(Y, old_parents, self.data)
                yield(operation, score_delta)

        for (X, Y) in model.edges():  # (3) flip single edge
            new_edges = model.edges() + [(Y, X)]
            new_edges.remove((X, Y))
            if nx.is_directed_acyclic_graph(nx.DiGraph(new_edges)):
                operation = ('flip', (X, Y))
                if operation not in tabu_list and ('flip', (Y, X)) not in tabu_list:
                    old_X_parents = model.get_parents(X)
                    old_Y_parents = model.get_parents(Y)
                    new_X_parents = old_X_parents + [Y]
                    new_Y_parents = old_Y_parents[:]
                    new_Y_parents.remove(X)
                    if max_indegree is None or len(new_X_parents) <= max_indegree:
                        score_delta = (local_score(X, new_X_parents, self.data) +
                                       local_score(Y, new_Y_parents, self.data) -
                                       local_score(X, old_X_parents, self.data) -
                                       local_score(Y, old_Y_parents, self.data))
                        yield(operation, score_delta)
    #--------------------------------------------------------------------------------------------------------------------
    """ Implementation of _legal_operations function for Dynamic Bayesian Networks 
        Contribution by Mariana C. Souza 
        References
        ---------
        [1] Nir Friedman Kevin Murphy Stuart Russell, Learning the Structure of Dynamic Probabilistic Networks, 1998 """
    #--------------------------------------------------------------------------------------------------------------------
    def _legal_operations_dynamic(self, model, tabu_list=[], max_indegree=None):
        """Generates a list of legal (= not in tabu_list) graph modifications
        for a given model, together with their score changes. Possible graph modifications:
        (1) add, (2) remove, or (3) flip a single edge. 
        If a number `max_indegree` is provided, only modifications that keep the number
        of parents for each node below `max_indegree` are considered."""

        local_score_dynamic = self.scoring_method.local_score_dynamic
        nodes = self.nodes
        potential_new_edges = (set(product(nodes, repeat=2)) - set([(X, Y) for ((X, A), (Y, B)) in model.edges()]))

        for (X, Y) in potential_new_edges:  # (1) add single edge
            if nx.is_directed_acyclic_graph(nx.DiGraph(model.edges() + [((X,0), (Y,1))])):
                operation = ('+', ((X,0), (Y,1)))
                if operation not in tabu_list:
                    old_parents = model.get_parents_dynamic(model, Y, 1)
                    new_parents = [] + old_parents
                    new_parents.append((X,0))
                    if max_indegree is None or len(new_parents) <= max_indegree:
                        score_delta = local_score_dynamic(Y, new_parents) - local_score_dynamic(Y, old_parents)
                        yield(operation, score_delta)

        for (X, Y) in potential_new_edges:
            if nx.is_directed_acyclic_graph(nx.DiGraph(model.edges() + [((X, 1), (Y, 1))])):
                operation = ('+', ((X, 1), (Y, 1)))
                if operation not in tabu_list and X != Y:
                    old_parents = model.get_parents_dynamic(model, Y, 1)
                    new_parents = [] + old_parents
                    new_parents.append((X, 1))
                    if max_indegree is None or len(new_parents) <= max_indegree:
                        score_delta = local_score_dynamic(Y, new_parents) - local_score_dynamic(Y, old_parents)
                        yield (operation, score_delta)

        for ((X,A),(Y,B)) in model.edges():  # (2) remove single edge
            operation = ('-', ((X,A), (Y,B)))
            if operation not in tabu_list:
                old_parents = model.get_parents_dynamic(model, Y, B)
                new_parents = old_parents[:]
                new_parents.remove((X,A))
                score_delta = local_score_dynamic(Y, new_parents) - local_score_dynamic(Y, old_parents)
                yield(operation, score_delta)

        for ((X,A),(Y,B)) in model.edges():  # (3) flip single edge
            if ((Y,A),(X,B)) not in model.edges():
                new_edges = model.edges() + [((Y,A), (X,B))]
                new_edges.remove(((X,A), (Y,B)))
                if nx.is_directed_acyclic_graph(nx.DiGraph(new_edges)):
                    operation = ('flip', ((X,A), (Y,B)))
                    if operation not in tabu_list and ('flip', ((Y,A), (X,B))) not in tabu_list:
                        old_X_parents = model.get_parents_dynamic(model, X, B)
                        old_Y_parents = model.get_parents_dynamic(model, Y, B)
                        new_X_parents = [] + old_X_parents
                        new_X_parents.append((Y,A))
                        new_Y_parents = old_Y_parents
                        new_Y_parents.remove((X,A))
                        if max_indegree is None or len(new_X_parents) <= max_indegree:
                            score_delta = (local_score_dynamic(X, new_X_parents) +
                                           local_score_dynamic(Y, new_Y_parents) -
                                           local_score_dynamic(X, old_X_parents) -
                                           local_score_dynamic(Y, old_Y_parents))
                            yield(operation, score_delta)


    def estimate(self, start=None, tabu_length=0, max_indegree=None):
        """
        Performs local hill climb search to estimates the `BayesianModel` structure
        that has optimal score, according to the scoring method supplied in the constructor.
        Starts at model `start` and proceeds by step-by-step network modifications
        until a local maximum is reached. Only estimates network structure, no parametrization.

        Parameters
        ----------
        start: BayesianModel instance
            The starting point for the local search. By default a completely disconnected network is used.
        tabu_length: int
            If provided, the last `tabu_length` graph modifications cannot be reversed
            during the search procedure. This serves to enforce a wider exploration
            of the search space. Default value: 100.
        max_indegree: int or None
            If provided and unequal None, the procedure only searches among models
            where all nodes have at most `max_indegree` parents. Defaults to None.

        Returns
        -------
        model: `BayesianModel` instance
            A `BayesianModel` at a (local) score maximum.

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> from pgmpy.estimators import HillClimbSearch, BicScore
        >>> # create data sample with 9 random variables:
        ... data = pd.DataFrame(np.random.randint(0, 5, size=(5000, 9)), columns=list('ABCDEFGHI'))
        >>> # add 10th dependent variable
        ... data['J'] = data['A'] * data['B']
        >>> labels = np.array(data.columns)
        >>> est = HillClimbSearch(data, labels, scoring_method=BicScore(data, labels))
        >>> best_model = est.estimate()
        >>> sorted(best_model.nodes())
        ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
        >>> best_model.edges()
        [('B', 'J'), ('A', 'J')]
        >>> # search a model with restriction on the number of parents:
        >>> est.estimate(max_indegree=1).edges()
        [('J', 'A'), ('B', 'J')]
        """
        epsilon = 1e-8
        if self.nodes is None:
            nodes = self.state_names.keys()
        else:
            nodes = self.nodes
        if start is None:
            start = BayesianModel()
            start.add_nodes_from(nodes)
        elif not isinstance(start, BayesianModel) or not set(start.nodes()) == set(nodes):
            raise ValueError("'start' should be a BayesianModel with the same variables as the data set, or 'None'.")

        tabu_list = []
        current_model = start

        while True:
            best_score_delta = 0
            best_operation = None

            for operation, score_delta in self._legal_operations(current_model, tabu_list, max_indegree):
                if score_delta > best_score_delta:

                    best_operation = operation
                    best_score_delta = score_delta
                    #print "best_operation: ", best_operation
            if best_operation is None or best_score_delta < epsilon:
                break
            elif best_operation[0] == '+':
                #print "best_operation choose: ", best_operation
                current_model.add_edge(*best_operation[1])
                tabu_list = ([('-', best_operation[1])] + tabu_list)[:tabu_length]
            elif best_operation[0] == '-':
                current_model.remove_edge(*best_operation[1])
                tabu_list = ([('+', best_operation[1])] + tabu_list)[:tabu_length]
            elif best_operation[0] == 'flip':
                X, Y = best_operation[1]
                current_model.remove_edge(X, Y)
                current_model.add_edge(Y, X)
                tabu_list = ([best_operation] + tabu_list)[:tabu_length]

        return current_model

    
    #--------------------------------------------------------------------------------------------------------------------
    """ Implementation of estimate function for Dynamic Bayesian Networks 
        Contribution by Mariana C. Souza 
        References
        ---------
        [1] Nir Friedman Kevin Murphy Stuart Russell, Learning the Structure of Dynamic Probabilistic Networks, 1998 """
    #--------------------------------------------------------------------------------------------------------------------
    def estimate_dynamic(self, start=None, tabu_length=0, max_indegree=None):
        """
        Performs local hill climb search to estimates the `BayesianModel` structure
        that has optimal score, according to the scoring method supplied in the constructor.
        Starts at model `start` and proceeds by step-by-step network modifications
        until a local maximum is reached. Only estimates network structure, no parametrization.

        Parameters
        ----------
        start: DynamicBayesianNetwork instance
            The starting point for the local search. By default a completely disconnected network is used.
        tabu_length: int
            If provided, the last `tabu_length` graph modifications cannot be reversed
            during the search procedure. This serves to enforce a wider exploration
            of the search space. Default value: 100.
        max_indegree: int or None
            If provided and unequal None, the procedure only searches among models
            where all nodes have at most `max_indegree` parents. Defaults to None.

        Returns
        -------
        model: `DynamicBayesianNetwork` instance
            A `DynamicBayesianModel` at a (local) score maximum.

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> from pgmpy.estimators import HillClimbSearch, BicScore
        >>> from pgmpy.models import DynamicBayesianNetwork as DBN
        >>> # create data sample with 9 random variables:
        ... data = pd.DataFrame(np.random.randint(0, 5, size=(5000, 9)), c   olumns=list('ABCDEFGHI'))
        >>> # add 10th dependent variable
        ... data['J'] = data['A'] * data['B']
        >>> labels = np.array(data.columns)
        >>> transitionModel = DBN()
        >>> transitionModel.add_nodes_from(labels)
        >>> est = HillClimbSearch(data, labels, scoring_method=BicScore(data, labels))
        >>> best_model = est.estimate_dynamic(start=transitionModel)
        >>> sorted(best_model.nodes())
        ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
        >>> best_model.edges()
        [(('A', 1), ('J', 1)), (('A', 1), ('B', 1)), (('J', 1), ('B', 1))]
        """

        epsilon = 1e-8
        if self.nodes is None:
            nodes = self.state_names.keys()
        else:
            nodes = self.nodes
        if start is None:
            start = DB()
            start.add_nodes_from(nodes)
        elif not isinstance(start, DB) or not set(start.nodes()) == set(nodes):
            raise ValueError("'start' should be a DynamicBayesianModel with the same variables as the data set, or 'None'.")

        tabu_list = []
        current_model = start

        while True:
            best_score_delta = 0
            best_operation = None

            for operation, score_delta in self._legal_operations_dynamic(current_model, tabu_list, max_indegree):
                if score_delta > best_score_delta:
                    best_operation = operation
                    best_score_delta = score_delta

            if best_operation is None or best_score_delta < epsilon:
                break
            elif best_operation[0] == '+':
                current_model.add_edges_from([(best_operation[1][0], best_operation[1][1])])
              
                tabu_list = ([('-', best_operation[1])] + tabu_list)[:tabu_length]
            elif best_operation[0] == '-':
                current_model.remove_edge((best_operation[1][0]),(best_operation[1][1]))
                tabu_list = ([('+', best_operation[1])] + tabu_list)[:tabu_length]
            elif best_operation[0] == 'flip':
                ((X,A), (Y,B)) = best_operation[1]
                current_model.remove_edge((X,A), (Y,B))
                current_model.add_edge((Y,A), (X,B))
                tabu_list = ([best_operation] + tabu_list)[:tabu_length]
        return current_model

