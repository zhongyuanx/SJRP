import csv
import numpy as np

# Create csv files for the problem parameters

dim = 12

mu_data = np.array([40.0, 35.0, 40.0, 40.0, 40.0, 20.0, 20.0, 20.0, 28.0, 20.0, 20.0, 20.0]) # annual demand rate

with open("configurations/12dim/mu_12dim.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(mu_data)

holding_cost_data = [2.0]*dim # holding cost
backlogging_cost_data = [100.0]*dim # backlogging cost

with open("configurations/12dim/inventory_cost_12dim.csv", "w") as file:
    writer = csv.writer(file)
    writer.writerow(holding_cost_data)
    writer.writerow(backlogging_cost_data)


fixed_cost_data = [100.0] # fixed ordering cost
variable_cost_data = [0.1, 0.1, 0.2, 0.2, 0.4, 0.2, 0.4, 0.4, 0.6, 0.6, 0.8, 0.8] # variable ordering cost


with open("configurations/12dim/ordering_cost_12dim.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(fixed_cost_data)
    writer.writerow(variable_cost_data)

CV = 0.5 # coefficient of variation of annual demand
sigma_data = CV*mu_data

with open("configurations/12dim/sigma_12dim.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(sigma_data)