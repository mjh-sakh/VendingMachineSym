import pandas as pd
import pytest
from functools import partial

from simulator.optimizers import *
from simulator.vmSim import *


# model set up
@pytest.fixture
def local_time() -> SimulationTime:
    return SimulationTime()


# -- Locations --
@pytest.fixture
def high_traffic_loc() -> Location:
    location = Location('railroad', traffic_conf_interval=(20, 40))
    return location


@pytest.fixture
def low_traffic_loc() -> Location:
    return Location('boring place', traffic_conf_interval=(3, 10))


# -- Products --
@pytest.fixture
def products() -> pd.DataFrame:
    return pd.DataFrame(columns=[
         'name',      'CI',    'size', 'cost', 'margin'], data=[
        ('papa_beer', (30, 50), 330,    250,     70),
        ('mama_beer', (10, 20), 330,    300,    100),
        ('baby_beer',  (1,  5), 250,    200,    150),
        ('water',     (15, 25), 500,    100,     50),
    ])


# -- VMs --
@pytest.fixture
def vm(high_traffic_loc, local_time):
    return VendingMachine(name='all_products', location=high_traffic_loc, time=local_time, columns=[
        ('papa_beer', 40),
        ('papa_beer', 40),
        ('mama_beer', 40),
        ('mama_beer', 40),
        ('baby_beer', 40),
        ('water', 30),
        ('water', 30),
    ])


# -- strategies --
@pytest.fixture
def test_strategy():
    return FillUpAllExistingToMaxOnMinLevel('test_one', min_levels={
        'papa_beer': 10,
        'mama_beer': 10,
        'baby_beer': 10,
        'water': 10,
    })


def calc_utility(sim: Simulation, *, interest_rate: float, trip_cost: float) -> float:
    expenses = sum(sim.refills_per_day) * trip_cost + np.mean(sim.total_inventory_cost) / 365 * len(
        sim.total_inventory_cost) * interest_rate
    profits = sum(sim.profit)[0]
    revenue = profits - expenses
    return revenue


def grid_for_products(grid: List, products: pd.DataFrame) -> dict[str, list]:
    return dict(zip(products.name.to_list(), [grid] * products.shape[1]))


@pytest.mark.parametrize('grid, extra, Updater, random_sampling, utility_func, interest_rate, trip_cost', [
    ([1, 20, 30], {'how_many_should_hit_min': [1]}, StrategyUpdater, 0.1, calc_utility, 0.03, 500),
    (['papa_beer mama_beer', 'papa_beer water'], {}, ColumnsUpdater, 1, calc_utility, 0.03, 500),
])
def test_grid_search_run(grid, extra, Updater, random_sampling, utility_func, interest_rate, trip_cost,
                         local_time, high_traffic_loc, products, vm, test_strategy):
    search_grid = {**grid_for_products(grid, products), **extra}
    if Updater is ColumnsUpdater:
        search_grid = {'columns': grid}
    sim = Simulation(
        'test_run',
        products=products,
        VMs=[vm],
        STGs={vm: test_strategy},
        local_time=local_time,
        cycles=100
    )

    updater = Updater(sim)

    gs = GridSearch(
        updater=updater,
        param_grid=search_grid,
        scoring_function=partial(utility_func, interest_rate=interest_rate, trip_cost=trip_cost),
        random_sampling=random_sampling
    )

    gs.calc_scores()
    gs.scores_matrix
    assert np.sum(gs.scores_matrix > 0) > 0
