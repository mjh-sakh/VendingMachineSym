# About
A MonteCarlo simulator of vending machines network that takes products popularity, locations traffic, vending machine inventory and refill strategies to simulate efficiency of operations. 

# Mechanics and assumptions

**Customer** preferences are generated based on products popularities. The popularity is provided by 90% confidence interval of lognormal distribution. So a single customer prefernce for the given product is sampled from its' distribution.  

Customer pick of the product is currently implemented in a following manner:
- availalbe products are checked for given vending machine
- preferences for available products are taken from customer
- selected preferences are normalized and put in the list
- random number generated betwee 0 and 1 and corresponding list item is selected
