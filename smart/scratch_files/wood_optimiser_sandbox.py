from pulp import *


# Creates a list of all demand nodes
clusters = [0, 1, 2]
wood_cluster_names = ["wood_1","wood_2","wood_3"]
worker_pos = [(0,0), (3,3), (6,6), (7,7)]
# Creates a list of all the supply nodes
workers = [0, 1, 2, 3]
# Worker distance to city in approximate move turns
distance = [[4, 5, 9],
            [5, 7, 6],
            [8, 1, 2],
            [4, 2, 5]]
# Creates the prob variable to contain the problem data
prob = LpProblem("Wood-exploitation-problem",LpMaximize)
# Creates a list of tuples containing all the possible routes for transport
Routes = [(w,c) for w in workers for c in clusters]
# A dictionary called route_vars is created to contain the referenced variables (the routes)
route_y = LpVariable.dicts("Route_y",(workers,clusters),0, cat="Binary")
# The objective function is added to prob first
prob += lpSum([route_y[w][c] for (w,c) in Routes])*20 - lpSum([distance[w][c] * route_y[w][c] for (w,c) in Routes]), "Sum new wood value vs worker distance"
# The supply maximum constraints are added to prob for each supply node (worker)
for w in workers:
    prob += lpSum([route_y[w][c] for c in clusters]) <= 1, "Max 1 job per worker %s"%w
# The demand minimum constraints are added to prob for each demand node (city)
for c in clusters:
    prob += lpSum([route_y[w][c] for w in workers]) <= 1, "Max 1 worker sent to each cluster %s"%c
prob += lpSum([route_y[w][c] for (w,c) in Routes]) <= len(workers)/2, "Max half workers assigned %s"%w
# Solve the optimization problem
p = prob.solve(PULP_CBC_CMD(msg=0, timeLimit=2))


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

orders = [[worker_pos[int(i[0])], wood_cluster_names[int(i[1])]] for i in orders]
