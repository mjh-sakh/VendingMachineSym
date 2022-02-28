from functools import reduce
from itertools import product
from typing import Any, Callable, List

import numpy as np
from ubelt import ProgIter

from strategies import FillUpAllExistingToMaxOnMinLevel
from vmSim import Simulation


class GridSearch:

    def __init__(self, *,
                 sim: Simulation,
                 param_grid: dict[str, Any],
                 scoring_function: Callable[[Simulation], float]) -> None:
        self.sim = sim
        self.param_grid = param_grid
        self.scoring_function = scoring_function
        self.param_names = sorted(param_grid.keys())
        self.scores: dict[str, float] = {}
        self.params_matrix = np.array([params for params in self]).reshape(self.shape)
        self.scores_matrix = np.zeros(self.shape[:-1] + [1])
        np.random.seed(321)

    @property
    def shape(self) -> List[int]:
        shape = []
        for name in self.param_names:
            shape.append(len(self.param_grid[name]))
        shape.append(len(self.param_names))
        return shape

    def __iter__(self):
        """Iterate over the points in the grid.
        Returns
        -------
        params : iterator over dict of str to any
            Yields dictionaries mapping each estimator parameter to one of its
            allowed values.
        """
        # Always sort the keys of a dictionary, for reproducibility
        items = sorted(self.param_grid.items())
        if not items:
            yield {}
        else:
            keys, values = zip(*items)
            for v in product(*values):
                params = dict(zip(keys, v))
                yield v

    def iter_indexes(self, shape, acc=[]):
        """
        iterate over matrix indices
        """
        if len(shape) == 1:
            yield tuple(acc)
        else:
            for i in range(shape[0]):
                yield from self.iter_indexes(shape[1:], acc=acc + [i])

    @staticmethod
    def hash_(params: List[Any]) -> str:
        return '-'.join(map(str, params))

    def score(self, params: List[Any]) -> float:
        """
        update simulation parameters and run scoring function on it
        """
        # score = self.scores.get(self.hash_(params))
        # if score:
        #     return score

        self.update_sim(params)
        self.sim.run()
        score = self.scoring_function(self.sim)
        # self.scores[self.hash_(params)] = score
        return score

    def update_sim(self, params: list[Any]) -> None:
        """
        update simulation parameters in order to run it again
        currently it's tailored for one particular one with min_levels and how_many_should_hit_min params
        """
        updated_strategy: FillUpAllExistingToMaxOnMinLevel = self.sim.STGs[self.sim.VMs[0]]
        params_dict = dict(zip(self.param_names, params))
        for name in updated_strategy.min_levels.keys():
            updated_strategy.min_levels[name] = params_dict.get(name) or updated_strategy.min_levels[name]
        if (value := params_dict['how_many_should_hit_min']):
            updated_strategy.how_many_should_hit_min = value
        self.sim.STGs[self.sim.VMs[0]] = updated_strategy

    def calc_scores(self) -> None:
        """
        go over entire grid of parameters and calculate score
        record score to scores_matrix with associated params_matrix
        """
        prog = ProgIter(desc='Grid Search', total=reduce(lambda i, acc: acc * i, self.scores_matrix.shape, 1),
                        verbose=1)
        prog.begin()
        for ix in self.iter_indexes(self.shape):
            score = self.score(self.params_matrix[ix])
            self.scores_matrix[ix] = score
            prog.step(inc=1)
        prog.end()


class SGD:
    def __init__(self, *,
                 sim: Simulation,
                 param_ranges: dict,
                 scoring_function: Callable[[Simulation], float]) -> None:
        self.sim = sim
        self.param_ranges = param_ranges
        self.scoring_function = scoring_function

    def calc_grad(self):
        pass
