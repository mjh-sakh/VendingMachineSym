from typing import Union, List
import numpy as np
import math
import random
import copy
from collections import defaultdict


def conf_interval_to_normal_distribution(minimum:float, maximum:float, z=1.645):
    """
    Conversts 90% confidence interval to mean and sigma for normal distribution
    z = 1.645  #for 90%
    z = 1.96  #for 95%
    z = 2.576  #for 99%
    """
    return minimum + (maximum - minimum)/2, (maximum - minimum) / (2*z)


def conf_interval_to_lognormal_distribution(minimum:float, maximum:float):
    """Conversts 90% confidence interval to mean and sigma for log-normal distribution"""
    return conf_interval_to_normal_distribution(math.log(minimum), math.log(maximum))


class SimulationTime:
    def __init__(self):
        self.today = 0
        
    def click(self):
        self.today += 1
        
    def reset(self):
        self.today = 0


class VendingMachine:
    """
    VendingMachine
    
    columns - dict[product] amount
    """
    def __init__(self, *, name, columns, location, time):
        self.name = name
        self.columns = copy.deepcopy(columns)
        self.location = location
        self.history = defaultdict(lambda: [])
        self.time = time
        self.sold_out_tag = 'Sold out'
        self.empty_tag = 'Empty'
    
    @property
    def available_products(self) -> List[str]:
        return [name for name, ammount in self.columns.items() if ammount > 0]
    
    def dispense_product(self, product_name):
        if product_name is None:
            self.write_history(self.empty_tag)
            return
        self.columns[product_name] -= 1
        self.write_history(product_name)
        if self.columns[product_name] == 0:
            self.write_history(self.sold_out_tag)
    
    @property
    def inventory(self):
        return self.columns
        # inventory = defaultdict(lambda: 0)
        # for name, count in self.columns:
        #     inventory[name] += count
        # return inventory
    
    def write_history(self, product_name):
        self.history[self.time.today].append(product_name)
        
    def refill(self, refill_data) -> dict:
        """
        recieves deltas for each product
        """
        for name, ammount in refill_data.items():
            self.columns[name] += ammount
            if self.columns[name] < 0: raise
            
    @property
    def today(self):
        return self.time.today
    
    @property
    def today_sales(self):
        sales = defaultdict(lambda: 0)
        for item in self.history[self.today]:
            sales[item] += 1
        return sales
    
    @property
    def today_sold_outs(self):
        sold_outs = 0
        for item in self.history[self.today]:
            if item in (self.sold_out_tag, self.empty_tag): sold_outs += 1
        return sold_outs
    
    def __hash__(self):
        return hash(f"{self.name}{self.location.name}")


class Location:
    """
    Location
    """
    def __init__(self, name: str, traffic_CI: tuple):
        self.name = name
        self.traffic_CI = traffic_CI
    
    @property
    def visits_today(self) -> int:
        return int(np.random.lognormal(*conf_interval_to_lognormal_distribution(*self.traffic_CI)))
        
    
class Customer:
    """
    Customer
    
    """
    def __init__(self, products):
        preferences = dict()
        for name, (CI_min, CI_max) in products.items():
            preferences[name] = (np.random.lognormal(*conf_interval_to_lognormal_distribution(CI_min, CI_max)))
        self.preferences = preferences
        
    def __getitem__(self, *names) -> List[float]:
        """
        return preferences for each name of product in the order it's provided
        """
        return [self.preferences[name] for name in names]
    
    def pick(self, choices: List[str]) -> str:
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
    
    
class BaseStrategy:
    def __init__(self, name):
        self.name = name
        
    def make_refil_decision(self, vending_machine: VendingMachine) -> Union[bool, dict]:
        """
        takes state of provided vending machine and makes decision to refill or not
        if refill, returns product-ammount dict
        if not, returns False
        """
        self.vm = vending_machine
        
        raise NotImplemented

        decision: Union(bool, dict) = False
        self.write_decision(decision)
        return decision

    
    @property
    def last_decision(self):
        return read_days_decision(self.vm, self.vm.today - 1)
    
    @classmethod
    def read_days_decision(vending_machine: VendingMachine, day: int) -> Union[bool, dict]:
        """
        reads previous decision from VM history
        """
        return vending_machine.get(f"decision_{day}")
        
    def write_decision(self, decision: Union[bool, dict]) -> None:
        """
        reads previous decision from VM history
        """
        self.vm.history[f"decision_{self.vm.today}"] = decision
