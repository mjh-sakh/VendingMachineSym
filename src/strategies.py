from vmSim import BaseStrategy, VendingMachine, Decision


class FillUpAllExistingToMaxOnMinLevel(BaseStrategy):

    def __init__(self, name: str, min_levels: dict[str, int], how_many_should_hit_min: int = 1):
        super().__init__(name)
        self.name = name
        self.min_levels = min_levels
        self.how_many_should_hit_min = how_many_should_hit_min

    def make_refill_decision(self, vending_machine: VendingMachine) -> Decision:
        """
        takes state of provided vending machine and makes decision to refill or not
        if refilled, returns product-amount dict
        if not, returns False
        """
        self.vm = vending_machine
        decision: Decision = {}
        hit_min = 0
        for name, amount in self.vm.inventory.items():
            decision[name] = self.vm.capacity[name]
            if amount < self.min_levels[name]:
                hit_min += 1
        if hit_min < self.how_many_should_hit_min:
            decision = False
        return decision
