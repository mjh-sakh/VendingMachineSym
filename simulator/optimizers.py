from functools import reduce
from itertools import product
from typing import Any, Callable, List
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from ubelt import ProgIter

from simulator.strategies import FillUpAllExistingToMaxOnMinLevel
from simulator.vmSim import Simulation


class UtilityFunctions:
    @staticmethod
    def revenue(sim: Simulation, *, interest_rate: float, trip_cost: float) -> float:
        expenses = sum(sim.refills_per_day) * trip_cost + np.mean(sim.total_inventory_cost) / 365 * len(
            sim.total_inventory_cost) * interest_rate
        profits = sum(sim.profit)[0]
        revenue = profits - expenses
        return revenue


class GridSearch:

    def __init__(self, *,
                 sim: Simulation,
                 param_grid: dict[str, Any],
                 scoring_function: Callable[[Simulation], float],
                 random_sampling: Union[float, int] = 1.0) -> None:
        """
        take Simulation, param grid and scoring function, calculate score for each point of the grid
        optional random sampling is fraction of points to be scored, default is 1 (100%)
        integer value can be provided instead of fraction to specify max number of sample points
        """
        self.sim = sim
        self.param_grid = param_grid
        if random_sampling > 1:
            self.sample_size = int(random_sampling)
        else:
            self.random_sampling = min(1.0, max(0.0, random_sampling))
            self.sample_size = None
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

    def score(self, params: List[Any]) -> float:
        """
        update simulation parameters and run scoring function on it
        """
        self.sim.local_time.reset()
        self.update_sim(params)
        self.sim.run()
        score = self.scoring_function(self.sim)
        return score

    def get_sample_points(self):
        grid_size = reduce(lambda i, acc: acc * i, self.scores_matrix.shape, 1)
        all_points = list(self.iter_indexes(self.shape))
        np.random.shuffle(all_points)
        self.sample_size = self.sample_size or int(grid_size * self.random_sampling) or 1
        return all_points[:self.sample_size]

    def calc_scores(self) -> None:
        """
        go over entire grid of parameters and calculate score
        record score to scores_matrix with associated params_matrix
        """
        sample_grid_points = self.get_sample_points()
        prog = ProgIter(desc='Grid Search', total=len(sample_grid_points), verbose=1)
        prog.begin()
        with ProcessPoolExecutor() as executor:
            for ix, score in zip(sample_grid_points, executor.map(self.score_for_ix, sample_grid_points)):
                self.scores_matrix[ix] = score
                prog.step(inc=1)
        prog.end()

    def score_for_ix(self, ix):
        return self.score(self.params_matrix[tuple(ix)])


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
