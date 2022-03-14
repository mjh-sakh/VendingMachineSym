from __future__ import annotations

import copy
import math
import random
from collections import defaultdict
from typing import Tuple, Union, List, Optional

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# -- type aliases --
Products = pd.DataFrame
ProductsCI = dict[str, Tuple[float, float]]
# columns = ['name', 'CI', 'size', 'cost', 'margin']
Decision = Union[bool, dict[str, int]]


def _lambda_zero():
    return 0


def _lambda_empty_list():
    return []


def conf_interval_to_normal_distribution(minimum: float, maximum: float, z: float = 1.645) -> Tuple[float, float]:
    """
    Convert 90% confidence interval to mean and sigma for normal distribution
    z = 1.645  #for 90%
    z = 1.96  #for 95%
    z = 2.576  #for 99%
    """
    return minimum + (maximum - minimum) / 2, (maximum - minimum) / (2 * z)


def conf_interval_to_lognormal_distribution(minimum: float, maximum: float) -> Tuple[float, float]:
    """Convert 90% confidence interval to mean and sigma for log-normal distribution"""
    return conf_interval_to_normal_distribution(math.log(minimum), math.log(maximum))


class SimulationTime:
    def __init__(self):
        self.today = 0

    def click(self):
        self.today += 1

    def reset(self):
        self.today = 0


class Product:
    """
    Product

    Not in use
    """

    def __init__(self, *,
                 name: str,
                 conf_interval: Tuple[int, int],
                 size: int,
                 cost: int,
                 margin: int) -> None:
        self.name = name
        self.CI = conf_interval
        self.size = size
        self.cost = cost
        self.margin = margin
        self.df: Union[None, pd.DataFrame] = None

    def as_df(self) -> pd.DataFrame:
        if self.df is None:
            self.df = pd.DataFrame({
                'name': self.name,
                'CI': self.CI,
                'size': self.size,
                'cost': self.cost,
                'margin': self.margin,
            })
        return self.df


class VendingMachine:
    """
    VendingMachine

    columns - dict[product] amount
    """
    Capacity = int
    ProductName = Optional[str]
    Column = Tuple[ProductName, Capacity]

    def __init__(self, *,
                 name: str,
                 columns: List[Column],
                 location: Location,
                 time: SimulationTime):
        self.name = name
        self.columns = copy.deepcopy(columns)
        self.holdings = [0] * len(columns)
        self.location = location
        self.history: dict[Union[int, str], Union[List[str], Decision]] = defaultdict(_lambda_empty_list)
        self.time = time
        self.sold_out_tag = 'Sold out'
        self.empty_tag = 'Empty'

    def __repr__(self):
        return self.name

    @property
    def available_products(self) -> List[ProductName]:
        return [name for name, amount in self.inventory.items() if amount > 0]

    def dispense_product(self, product_name: Optional[Union[bool, str]]):
        if product_name is None:
            self.write_history(self.empty_tag)
            return
        if product_name is False:
            self.write_history(self.sold_out_tag)
            return
        self.inventory = (product_name, self.inventory[product_name] - 1)
        self.write_history(product_name)
        if self.inventory[product_name] == 0:
            self.write_history(self.sold_out_tag)

    @property
    def capacity(self) -> defaultdict[ProductName, int]:
        capacity = defaultdict(_lambda_zero)
        for name, column_capacity in self.columns:
            capacity[name] += column_capacity
        return capacity

    @property
    def inventory(self) -> defaultdict[ProductName, int]:
        inventory = defaultdict(_lambda_zero)
        for i, (name, _) in enumerate(self.columns):
            inventory[name] += self.holdings[i]
        return inventory

    @inventory.setter
    def inventory(self, name_value: Tuple[ProductName, int]):
        name, value = name_value
        for column_ix, (prod_name, capacity) in enumerate(
                self.columns):
            if name == prod_name:
                self.holdings[column_ix] = min(capacity, value)
                value = max(0, value - capacity)

    def write_history(self, product_name: str) -> None:
        self.history[self.time.today].append(product_name)

    def refill(self, refill_data: Decision) -> None:
        """
        receive deltas for each product
        it will ignore overfill
        """
        for name, amount in refill_data.items():
            self.inventory = (name, amount)

    @property
    def today(self):
        return self.time.today

    @property
    def today_sales(self) -> defaultdict[str, int]:
        sales: defaultdict[str, int] = defaultdict(_lambda_zero )
        for item in self.history[self.today]:
            if item not in [self.sold_out_tag, self.empty_tag]:
                sales[item] += 1
        return sales

    @property
    def today_sold_outs(self):
        sold_outs = 0
        for item in self.history[self.today]:
            if item in (self.sold_out_tag, self.empty_tag):
                sold_outs += 1
        return sold_outs

    def __hash__(self):
        return hash(f"{self.name}{self.location.name}")


class Location:
    """
    Location
    """

    def __init__(self, name: str, traffic_conf_interval: Tuple[int, int]):
        self.name = name
        self.traffic_CI = traffic_conf_interval

    @property
    def visits_today(self) -> int:
        return int(np.random.lognormal(*conf_interval_to_lognormal_distribution(*self.traffic_CI)))


class Customer:
    """
    Customer

    """

    def __init__(self, products: ProductsCI):
        preferences: dict[str, float] = dict()
        for name, (CI_min, CI_max) in products.items():
            preferences[name] = np.random.lognormal(
                *conf_interval_to_lognormal_distribution(CI_min, CI_max))
        self.preferences = preferences
        self.preferences = self.get_normalized_preferences()
        self.willingness_to_retry = np.random.uniform(
            0.05, 0.25)  # some magic numbers for now

    def get_normalized_preferences(self, names: Optional[List[str]] = None) -> dict[str, float]:
        if names is None:
            names = list(self.preferences.keys())
        selected_values = [self.preferences[name] for name in names]
        _sum = sum(selected_values)
        return dict(zip(names, [value / _sum for value in selected_values]))

    def __getitem__(self, *names: str) -> List[float]:
        """
        return preferences for each name of product in the order it's provided
        """
        return [self.preferences[name] for name in names]

    def pick(self, choices: List[str]) -> Union[bool, str]:
        """
        take list of available products, do pick for all preferences and check if available
        if not available, do another try with some probability
        if not available again, then return False
        """
        current_choice = self.choose_one()
        if current_choice in choices:
            return current_choice
        if random.random() >= self.willingness_to_retry:
            names = [name for name in self.preferences.keys() if name !=
                     current_choice]
            current_choice = self.choose_one(names)
            if current_choice in choices:
                return current_choice
        return False

    def choose_one(self, names: Optional[List[str]] = None) -> str:  # slow
        if names is None:
            choices = self.preferences
        else:
            choices = self.get_normalized_preferences(names)
        probabilities = list(choices.values())
        choices_names = list(choices.keys())
        return np.random.choice(choices_names, p=probabilities)  # slow likely because of that

    def pick_old(self, choices: List[str]) -> str:
        """
        takes list of available products to be picked from and returns one
        """
        dist = np.array(self.__getitem__(*choices))
        normalized_dist = dist / sum(dist)
        coin_flip = random.random()
        accumulator = 0
        for i, probability in enumerate(normalized_dist):
            accumulator += probability
            if accumulator > coin_flip:
                return choices[i]


class Simulation:
    def __init__(self, name: str, *,
                 products: Products,
                 VMs: List[VendingMachine],
                 STGs: dict[VendingMachine, BaseStrategy],
                 cycles: int,
                 local_time: SimulationTime
                 ):
        self.name = name
        self.local_time = local_time
        self.products: ProductsCI = self.from_df_to_dict(products, ['name', 'CI'])
        self.product_costs = self.from_df_to_dict(products, ['name', 'cost'])
        self.product_margins = self.from_df_to_dict(
            products, ['name', 'margin'])
        self.VMs = VMs
        self.STGs = STGs
        self.cycles = cycles

        # -- stats --
        self.total_inventory_levels = pd.DataFrame()
        self.total_sales = pd.DataFrame()
        self.refills_per_day: List[int] = []
        self.sold_outs_per_day: List[int] = []
        self.total_inventory_cost = np.array([])
        self.profit = np.array([])

    @staticmethod
    def from_df_to_dict(df: pd.DataFrame, columns: List[str]):
        return dict(df[columns].to_dict('split')['data'])

    def run(self):
        total_inventory_levels = []
        total_sales = []
        self.refills_per_day = []
        self.sold_outs_per_day = []

        # initial fill
        for vm in self.VMs:
            refill_strategy = self.STGs[vm]
            refill_data = refill_strategy.make_refill_decision(vm)
            if refill_data:
                vm.refill(refill_data)

        for day in range(self.cycles):
            self.local_time.click()
            today_inventory_levels: defaultdict[str, int] = defaultdict(_lambda_zero)
            today_sales: defaultdict[str, int] = defaultdict(_lambda_zero)
            refills_count = 0
            sold_outs_count = 0

            for vm in self.VMs:
                self.complete_day_cycle(vm)
                today_inventory_levels['day'] = today_sales['day'] = self.local_time.today
                for name, amount in vm.inventory.items():
                    today_inventory_levels[name] += amount
                refill_strategy = self.STGs[vm]
                refill_data = refill_strategy.make_refill_decision(vm)
                if refill_data:
                    vm.refill(refill_data)
                    refills_count += 1
                for name, amount in vm.today_sales.items():
                    today_sales[name] += amount
                sold_outs_count += vm.today_sold_outs

            total_inventory_levels.append(today_inventory_levels)
            total_sales.append(today_sales)
            self.refills_per_day.append(refills_count)
            self.sold_outs_per_day.append(sold_outs_count)

        self.total_inventory_levels = pd.DataFrame(total_inventory_levels)
        self.total_inventory_levels.set_index('day', inplace=True)
        self.total_inventory_levels.fillna(0, inplace=True)
        self.total_sales = pd.DataFrame(total_sales)
        self.total_sales.set_index('day', inplace=True)
        self.total_sales.fillna(0, inplace=True)
        self.calc_stats()

    def calc_stats(self):
        self.total_inventory_cost = np.dot(self.total_inventory_levels[self.products.keys()].values,
                                           np.array([self.product_costs[product] for product in
                                                     self.products.keys()]).reshape(-1, 1))
        # this to make sure columns aligned, not just .values
        self.profit = np.dot(self.total_sales[self.products.keys()].fillna(0).values,
                             np.array([self.product_margins[product] for product in self.products.keys()]).reshape(-1,
                                                                                                                   1))
        # this to make sure columns aligned, not just .values

    def complete_day_cycle(self, vm: VendingMachine) -> None:
        for i in range(vm.location.visits_today):
            c = Customer(self.products)
            picked_product = c.pick(vm.available_products)
            vm.dispense_product(picked_product)

    def plot_stat(self, stat_name: str):
        plt.figure(figsize=(12, 5))
        plt.grid()
        sns.lineplot(data=self.__dict__[stat_name])
        plt.title(
            f'{stat_name.replace("_", " ").capitalize()} for "{self.name}" simulation')
        plt.show()


class BaseStrategy:
    def __init__(self, name: str):
        self.name = name
        self.vm: VendingMachine = None

    def __repr__(self):
        return self.name

    def make_refill_decision(self, vending_machine: VendingMachine) -> Decision:
        """
        takes state of provided vending machine and makes decision to refill or not
        if refilled, returns product-amount dict
        if not, returns False
        """
        self.vm = vending_machine

        raise NotImplemented

        decision: Decision = False
        self.write_decision(decision)
        return decision

    @property
    def last_decision(self) -> Decision:
        return self.read_days_decision(self.vm.today - 1)

    def read_days_decision(self, day: int) -> Decision:
        """
        reads previous decision from VM history
        """
        return self.vm.history.get(f"decision_{day}")

    def write_decision(self, decision: Decision) -> None:
        """
        reads previous decision from VM history
        """
        self.vm.history[f"decision_{self.vm.today}"] = decision
