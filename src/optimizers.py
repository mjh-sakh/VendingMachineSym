from itertools import product
from typing import Any, Callable, List

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

    @staticmethod
    def hash_(params: List[Any]) -> str:
        return '-'.join(map(str, params))

    def score(self, params: List[Any]) -> float:
        score = self.scores.get(self.hash_(params))
        if score:
            return score

        self.update_sim(params)
        self.sim.run()
        score = self.scoring_function(self.sim)
        self.scores[self.hash_(params)] = score
        return score

    def update_sim(self, params: list[Any]) -> None:
        updated_strategy: FillUpAllExistingToMaxOnMinLevel = self.sim.STGs[self.sim.VMs[0]]
        params_dict = dict(zip(self.param_names, params))
        for name in updated_strategy.min_levels.keys():
            updated_strategy.min_levels[name] = params_dict.get(name) or updated_strategy.min_levels[name]
        if (value := params_dict['how_many_should_hit_min']):
            updated_strategy.how_many_should_hit_min = value

    def calc_scores(self):
        for params in self:
            self.score(params)


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
