from pulp import *

city_ids = ["1", "2", "3"]
worker_pos = [(0,0), (1,1), (2,2), (2,3)]

# Creates a list of all the supply nodes
workers = [0, 1, 2, 3]

# Creates a list for the number of units of supply for each supply node
supply = [10, 500, 200, 10]

# Does worker supply contain special resources
supply_special = [0, 9, 0, 9]

# Creates a list of all demand nodes
cities = [0, 1, 2]

# Creates a list for the number of units of demand for each demand node
demand = [200, 700, 300]

# City value based on the number of city tiles
value = [30, 60, 10]

# Worker distance to city in approximate move turns
distance = [[4, 5, 9],
            [5, 7, 6],
            [8, 1, 2],
            [7, 4, 12]]


# Creates the prob variable to contain the problem data
prob = LpProblem("Fuel delivery problem",LpMaximize)

# Creates a list of tuples containing all the possible routes for transport
Routes = [(w,c) for w in workers for c in cities]
cities_y = {i: LpVariable(name=f"y{i}", cat="Binary") for i in range(len(cities))}

# A dictionary called route_vars is created to contain the referenced variables (the routes)
route_vars = LpVariable.dicts("Route",(workers,cities),0,None,LpInteger)
route_y = LpVariable.dicts("Route_y",(workers,cities),0, cat="Binary")

# The objective function is added to prob first
prob += lpSum([value[c] * cities_y[c] for c in cities]) - lpSum([distance[w][c] * route_y[w][c] for (w,c) in Routes]) + lpSum([supply_special[w] * route_y[w][c] for (w,c) in Routes]), "Sum of city value vs transporting cost"

# The supply maximum constraints are added to prob for each supply node (worker)
for w in workers:
    prob += lpSum([route_vars[w][c] for c in cities]) <= (supply[w]), "Sum of fuel out of worker %s"%w
    prob += lpSum([route_y[w][c] for c in cities]) <= 1, "workers y contraint %s"%w
    for c in cities:
        prob += route_vars[w][c] <= route_y[w][c] * 100000, f"limit workers 1 {w} {c}"
        prob += route_vars[w][c] >= route_y[w][c], f"limit workers 2 {w} {c}"

# The demand minimum constraints are added to prob for each demand node (city)
for c in cities:
    prob += lpSum([route_vars[w][c] for w in workers]) >= (demand[c] * cities_y[c]), "Sum of fuel into cities %s"%c



# Solve the optimization problem
status = prob.solve()
for var in prob.variables():
    if "Route_y_" in var.name and var.value() == 1:
        var.name[8:].split("_")


print(f"status: {prob.status}, {LpStatus[prob.status]}")
print(f"objective: {prob.objective.value()}")

for var in prob.variables():
    print(f"{var.name}: {var.value()}")

for name, constraint in prob.constraints.items():
    print(f"{name}: {constraint.value()}")


orders = []
for var in prob.variables():
    if "Route_y_" in var.name and var.value() == 1:
        orders.append(var.name[8:].split("_"))

orders = [[worker_pos[int(i[0])], city_ids[int(i[1])]] for i in orders]
[int(i[1]) for i in orders]
