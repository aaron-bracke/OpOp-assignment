import numpy as np
import random
from gurobipy import Model, GRB, quicksum
from itertools import combinations

# TODO: Didn't implement constraint 1t (linked to parameter f) yet
# TODO: Didn't implement constraint 1u (linked to parameter h) yet
# TODO: V&V

# Set random seed
random.seed(29112023)

# Parameter values
N_V = 3                                 # Number of pickup nodes
N_K = 5                                 # Number of garbage trucks
N_W = 1                                 # Number of types of waste
N_P = 1                                 # Number of sorting units
N_C = 2                                 # Number of types of garbage trucks

x_map_min, x_map_max = 0.0, 100.0       # The range of the map in x-direction
y_map_min, y_map_max = 0.0, 100.0       # The range of the map in y-direction

D_min, D_max = 10.0, 50.0       # Minimum and maximum amount of waste to be collected of a type if that type of waste is produced by the node

E_0w_min, E_0w_max = 4.0, 7.0   # Minimum and maximum earliest time for leaving the depot
E_0w = np.array([random.uniform(E_0w_min, E_0w_max) for w in range(N_W)])    # Array of earliest times for leaving the depot for the different wastes

L_0w_min, L_0w_max = 6.0, 9.0          # Minimum and maxmimum latest time for leaving the depot
L_0w = np.array([random.uniform(max(L_0w_min, E_0w[w]), L_0w_max) for w in range(N_W)])    # Array of latest times for leaving the depot for the different wastes

S_min, S_max = 0.1, 0.5         # Minimum and maximum time needed to pick up waste
Q_min, Q_max = 1000.0, 2000.0       # Minimum and maximum capacity for a garbage truck
P_min, P_max = 15.0, 17.0       # Minimum and maximum latest time for garbage trucks to be returned to the depot TODO: Check whether it's the time of departure to the depot or time of arrival

Alpha = 100.0           # Fixed cost of using garbage truck
Beta = 1.0              # Variable cost of garbage truck
Delta = 2.0             # Additional dwell time cost of garbage truck

Gamma1 = 0.0001         # Coefficient of fixed cost of using a garbage truck
Gamma2 = 0.0001         # Coefficient of variable cost of using a garbage truck
Gamma3 = 0.0001         # Coefficient of cost of extra dwelling of a garbage truck

M = 100.0               # Big M TODO: Find a good value for this

class Depot:
    def __init__(self):
        self.x = random.uniform(x_map_min, x_map_max)   # The x-coordinate of the depot
        self.y = random.uniform(y_map_min, y_map_max)   # The y-coordinate of the depot
        self.wastes = np.array([Waste(w, 0) for w in range(N_W)])   # An array containing all the wastes of the node

class Node:
    def __init__(self):
        self.x = random.uniform(x_map_min, x_map_max)   # The x-coordinate of the node
        self.y = random.uniform(y_map_min, y_map_max)   # The y-coordinate of the node
        self.wastes = np.array([Waste(w) for w in range(N_W)])      # An array containing all the wastes of the node

class SortingUnit:
    def __init__(self):
        self.x = random.uniform(x_map_min, x_map_max)   # The x-coordinate of the sorting unit
        self.y = random.uniform(y_map_min, y_map_max)   # The y-coordinate of the sorting unit
        self.wastes = np.array([Waste(w, 0) for w in range(N_W)])   # An array containing all the wastes of the node
    def SetPosition(self, x, y):
        self.x = x
        self.y = y

class Waste:
    def __init__(self, waste_type: int, amount = None):
        self.waste_type = waste_type                    # The type of this waste
        if amount == None:
            self.amount = random.uniform(D_min, D_max)  # The amount of this waste at its node
        else:
            self.amount = amount
        self.time_needed = random.uniform(S_min, S_max) # The time needed to pick up the waste at its node
        self.earliest_time = random.uniform(E_0w[self.waste_type], L_0w[self.waste_type] - self.time_needed)    # The earliest time of picking up this waste at its node
        self.latest_time = random.uniform(self.earliest_time + self.time_needed, L_0w[self.waste_type])         # The latest time of picking up this waste at its node        

class Truck:
    def __init__(self):
        # self.truck_type = truck_type                            # The truck type (which waste it can take)
        self.capacity = random.uniform(Q_min, Q_max)            # The capacity of the truck
        self.fixed_cost = Alpha                                 # The fixed cost of the garbage truck when it is used
        self.variable_cost = Beta                               # The variable cost of the garbage truck
        self.additional_cost = Delta                            # The additional dwell time cost
        self.latest_return_time = random.uniform(P_min, P_max)  # The latest time for the garbage truck to get back to its home depot

def TimeBetweenNodes(node1: Node, node2: Node):
    return np.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2)     # Assuming time between nodes is directly related to distance

### Create instances ###
# Create all the nodes
depot = Depot()                                                         # Depot
pickup_nodes = np.array([Node() for i in range(N_V)])                   # Set R
sorting_units = np.array([SortingUnit() for i in range(N_P * N_K)])     # Set P
for i in range(N_P):
    sorting_units[i * N_K + 1].SetPosition(sorting_units[i * N_K].x, sorting_units[i * N_K].y)
    sorting_units[i * N_K + 2].SetPosition(sorting_units[i * N_K].x, sorting_units[i * N_K].y)
nodes = np.hstack((depot, pickup_nodes, sorting_units))                 # Set V
# Determine times corresponding to arcs
t = {}  # Dict containing arc times
for i in range(len(nodes)):
    for j in range(len(nodes)):
        t[(nodes[i], nodes[j])] = TimeBetweenNodes(nodes[i], nodes[j])
# Create all the garbage trucks
trucks = np.array([Truck() for i in range(N_K)])
# Create the waste types
waste_types = range(N_W)

### Set up model ###
# Instantiate VRP model
VRP_model = Model("MSWSCRP")
# Add the decision variables to the model
x = VRP_model.addVars(nodes, nodes, trucks, waste_types,    vtype=GRB.BINARY,       name="x")   # 1 if garbage truck k collects solid waste w and traverses arc (i, j), 0 otherwise
y = VRP_model.addVars(trucks, waste_types,                  vtype=GRB.BINARY,       name="y")   # 1 if garbage truck k collects solid waste w, 0 otherwise
a = VRP_model.addVars(nodes, trucks,                        vtype=GRB.CONTINUOUS,   name="a")   # arrival time of garbage truck k to node i
u = VRP_model.addVars(nodes, trucks,                        vtype=GRB.CONTINUOUS,   name="u")   # dwell time of garbage truck k in node i

### Add the constraints ###
# Constraint 1b
VRP_model.addConstrs((quicksum(x[i, j, k, w] for k in trucks for j in nodes if j != i)  <= 1 
    for i in nodes[1:] 
    for w in waste_types), name="1b")
# Constraint 1c
VRP_model.addConstrs((quicksum(x[depot, j, k, w] for j in nodes[1:]) == y[k, w]
    for k in trucks 
    for w in waste_types), name="1c")
# Constraint 1d
VRP_model.addConstrs((quicksum(x[i, depot, k, w] for i in sorting_units) == y[k, w]
    for k in trucks 
    for w in waste_types), name="1d")
# Constraint 1e
VRP_model.addConstrs((quicksum(x[i, j, k, w] for i in nodes if i!=j) == quicksum(x[j, i, k, w] for i in nodes if i!=j)
    for j in nodes
    for k in trucks
    for w in waste_types), name="1e")
# Constraint 1f
VRP_model.addConstrs((x[i, j, k, w] + x[j, i, k, w] <= 1 for i in nodes[1:] 
    for j in nodes 
    for k in trucks 
    for w in waste_types if i!=j), name="1f")
# Constraint 1g 
VRP_model.addConstrs((x[i, i, k, w] == 0 
    for i in nodes 
    for k in trucks 
    for w in waste_types), name="1g")
# Constraint 1h
M_h = M # Big M for constraint 1h   TODO: determine a good value for M_h
VRP_model.addConstrs((quicksum(x[i, j, k, w] for j in nodes[1:] for i in nodes[1:] if i!=j) <= M_h * y[k, w]
    for k in trucks 
    for w in waste_types), name="1h")
# Constraint 1i
VRP_model.addConstrs((quicksum(i.wastes[w].amount *  x[i, j, k, w] for j in nodes for i in nodes[1:]) <= k.capacity # FIXME: Looks like there's a mistake in the paper here so idk
    for k in trucks
    for w in waste_types), name="1i")
# Constraint 1j
M_j = M # Big M for constraint 1j   TODO: determine a good value for M_j
VRP_model.addConstrs((a[i, k] + u[i, k] + quicksum(i.wastes[w].time_needed * y[k, w] for w in waste_types) + t[(i, j)] - a[j, k] <= M_j * (1 - quicksum(x[i, j, k, w] for w in waste_types))
    for i in nodes
    for j in nodes[1:]
    for k in trucks if i!=j), name="1j")
# Constraint 1k
M_k = M # Big M for constraint 1k   TODO: determine a good value for M_k
VRP_model.addConstrs((a[j, k] - u[i, k] - quicksum(i.wastes[w].time_needed * y[k, w] for w in waste_types) - t[(i, j)] - a[i, k] <= M_k * (1 - quicksum(x[i, j, k, w] for w in waste_types))
    for i in nodes
    for j in nodes[1:]
    for k in trucks if i!=j), name="1k")
# Constraint 1l
VRP_model.addConstrs((quicksum(i.wastes[w].earliest_time * quicksum(x[i, j, k, w] for j in nodes if i!=j) for w in waste_types) <= a[i, k] + u[i, k]
    for i in nodes[1:]
    for k in trucks), name="1l")
# Constraint 1m
VRP_model.addConstrs((a[i, k] + u[i, k] <= quicksum(i.wastes[w].latest_time * quicksum(x[i, j, k, w] for j in nodes if i!=j) for w in waste_types)
    for i in nodes[1:]
    for k in trucks), name="1m")
# Constraint 1n
M_n = M # Big M for constraint 1n   TODO: determine a good value for M_n
VRP_model.addConstrs((E_0w[w] - a[nodes[0], k] - u[nodes[0], k] <= M_n * (1 - y[k, w])
    for k in trucks
    for w in waste_types), name="1n")
# Constraint 1o
M_o = M # Big M for constraint 1o   TODO: determine a good value for M_o
VRP_model.addConstrs((a[nodes[0], k] + u[nodes[0], k] - L_0w[w] <= M_o * (1 - y[k, w])
    for k in trucks
    for w in waste_types), name="1o")
# Constraint 1p
VRP_model.addConstrs((a[i, k] + u[i, k] + quicksum(i.wastes[w].time_needed * y[k, w] for w in waste_types) <= k.latest_return_time
    for k in trucks
    for i in nodes[1:]), name="1p")
# Constraint 1q
VRP_model.addConstrs((quicksum(x[i, j, k, w] for i in nodes for j in sorting_units if i!=j) + quicksum(x[j, i, k, w] for i in nodes for j in sorting_units if i!=j) == 2 * y[k, w]
    for k in trucks
    for w in waste_types), name="1q")
# Constraint 1r
VRP_model.addConstrs((quicksum(x[i, j, k, w] for i in pickup_nodes for j in sorting_units) >= y[k, w]
    for k in trucks
    for w in waste_types), name="1r")
# Constraint 1s
VRP_model.addConstrs((quicksum(y[k, w] for w in waste_types) <= 1
    for k in trucks), name="1s")
# Constraint 1t
# TODO: Figure out what to do with this constraint
# VRP_model.addConstrs((x[i, j, k, w] <= f[i, j, k]
#     for i in nodes[1:]
#     for j in nodes[1:]
#     for k in trucks
#     for w in waste_types
#     if i!=j), name="1t")
# Constraint 1u
# TODO: Figure out what to do with this constraint
# VRP_model.addConstrs((y[k, w] <= h[k, w]
#     for k in trucks
#     for w in waste_types), name="1u")
# Constraint 1v
VRP_model.addConstrs((a[nodes[0], k] == 0
    for k in trucks), name="1v")
# Constraint 1w
VRP_model.addConstrs((a[i, k] >= 0
    for k in trucks
    for i in nodes), name="1w")
# Constraint 1x
VRP_model.addConstrs((u[i, k] >= 0
    for k in trucks
    for i in nodes), name="1x")
# Constraint 1y
# Binary decision variables constraint is satisfied during initialisation
# Constraint 1z
# Binary decision variables constraint is satisfied during initialisation

### Set the objective ###
VRP_model.setObjective(     quicksum((1 - y[k, w]) for k in trucks for w in waste_types)
                       +    quicksum(x[i, j, k, w] for i in nodes for j in nodes for k in trucks for w in waste_types)
                       -    Gamma1 * quicksum((k.fixed_cost * x[nodes[0], j, k, w]) for k in trucks for j in nodes[1:] for w in waste_types)
                       -    Gamma2 * quicksum((k.variable_cost * x[i, j, k, w]) for k in trucks for i in nodes[1:] for j in nodes[1:] for w in waste_types)
                       -    Gamma3 * quicksum((k.additional_cost * u[i, k]) for k in trucks for i in nodes[1:])
                       , GRB.MAXIMIZE)

### Display model information ###
VRP_model.update()
print("\nModel: ", end="")
VRP_model.printStats()
print("\n")

## Optimise the model ###
VRP_model.optimize()
