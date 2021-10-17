import numpy as np
from utils.base_controller import BaseController
from agent_objects.map_controller import MapController
from pulp import LpMaximize, LpProblem, LpVariable, LpInteger, LpStatus, lpSum, PULP_CBC_CMD
from lux.game import Game
from typing import Any, List
import networkx as nx
import logging

logging.basicConfig(filename="botlog.txt",
                    filemode='w',
                    level=logging.INFO)

class CityController(BaseController):
    def __init__(self, game_state: Game, observation: Any, mc: MapController):
        super().__init__(game_state, observation)
        self.mc = mc
        self.available_workers = self.get_available_workers()
        self.available_workers_positions = self.available_workers.get("positions")
        self.available_workers_supply = self.available_workers.get("supply")
        self.available_workers_special = self.available_workers.get("supply_special")
        self.available_workers_index = self.available_workers.get("index")
        self.cities_needing_fuel = self.get_cities_needing_fuel()
        self.cities_needing_fuel_id = self.cities_needing_fuel.get("city_id")
        self.cities_needing_fuel_demand = self.cities_needing_fuel.get("demand")
        self.cities_needing_fuel_value = self.cities_needing_fuel.get("value")
        self.cities_needing_fuel_index = self.cities_needing_fuel.get("index")
        self.cities_needing_fuel_turns = self.cities_needing_fuel.get("fuel_turns")
        self.worker_distance_to_cities = self.get_worker_distance_to_cities()
        self.fuel_distribution_orders = self.find_optimal_fuel_distribution()

    def use_cities(self, actions):
        new_workers = 0
        new_research = 0
        # iterate over cities and do something
        for k, city in self.player.cities.items():
            for citytile in city.citytiles:
                if citytile.can_act():
                    if (self.no_workers + new_workers) < self.no_city_tiles:
                        actions.append(citytile.build_worker())
                        new_workers += 1
                    
                    elif (self.player.research_points + new_research) < 200:
                        actions.append(citytile.research())
                        new_research += 1

    def get_available_workers(self) -> dict:
        positions = []
        supply = []
        supply_special = []
        index = []
        for unit in self.player.units:
            if unit.cargo.wood + (unit.cargo.coal*10) + (unit.cargo.uranium*40) >= 20:
                positions.append((unit.pos.x, unit.pos.y))
                supply.append(unit.cargo.wood + (unit.cargo.coal*10) + (unit.cargo.uranium*40))
                index.append((max(index)+1 if index else 0))
                if (unit.cargo.coal*10) + (unit.cargo.uranium*40) > 30:
                    supply_special.append(0.1)
                else:
                    supply_special.append(1)
        
        return {
            "positions": [positions[i] for i in np.argsort(supply)[::-1]][:40],
            "supply": [supply[i] for i in np.argsort(supply)[::-1]][:40],
            "supply_special": [supply_special[i] for i in np.argsort(supply)[::-1]][:40],
            "index": [index[i] for i in np.argsort(supply)[::-1]][:40],
            }
    
    def get_cities_needing_fuel(self) -> dict:
        city_id = []
        demand = []
        fuel_turns = []
        value = []
        index = []
        for city in self.player.cities.values():
            fuel = city.fuel
            upkeep = city.get_light_upkeep()
            
            if fuel < (upkeep * min((10 + (len(city.citytiles)/5)), 15)):
                city_id.append(f"{city.cityid}")
                demand.append(max((upkeep * min((10 + (len(city.citytiles)/5)), 15)) - fuel, 100))
                value.append(max(len(city.citytiles)/2, 4))
                index.append((max(index)+1 if index else 0))
                fuel_turns.append(int(fuel/upkeep))
        return {
            "city_id": [city_id[i] for i in np.argsort(value)[::-1]][:20],
            "demand": [demand[i] for i in np.argsort(value)[::-1]][:20],
            "value": [value[i] for i in np.argsort(value)[::-1]][:20],
            "index": [index[i] for i in np.argsort(value)[::-1]][:20],
            "fuel_turns": [fuel_turns[i] for i in np.argsort(value)[::-1]][:20],
            }
    
    def get_worker_distance_to_cities(self) -> np.ndarray:
        no_w = len(self.available_workers_positions)
        no_c = len(self.cities_needing_fuel_id)
        wp = self.available_workers_positions
        dist_matrix = np.ones(shape=(no_w, no_c))
        for w in range(len(wp)):
            dist_dic = self.find_distance_to_cities(source=wp[w])
            for c in dist_dic.items():
                c_name = c[0]
                c_dist = c[1]
                cid = self.cities_needing_fuel_id.index(c_name) if c_name in self.cities_needing_fuel_id else "No match"
                if cid != "No match":
                    if c_dist <= (self.turns_until_night):
                        dist_matrix[w][cid] = c_dist
                    elif ((c_dist - self.turns_until_night) * 2) > self.cities_needing_fuel_turns[cid]:
                        dist_matrix[w][cid] = 99999
                    else:
                        dist_matrix[w][cid] = self.turns_until_night + ((c_dist - self.turns_until_night) * 2)

        return dist_matrix

        # find closest route to node type
    def find_distance_to_cities(self, source) -> dict:
        G = self.mc.graph_map
        #Calculate the length of paths from source to all other nodes
        lengths=nx.single_source_dijkstra_path_length(G, source, weight="weight_ac")
        #We are only interested in a particular type of node
        subnodes = [name for name, d in G.nodes(data=True) if ("friendly_city" in d['type'])]
        subnode_type = [d['type'][1] for name, d in G.nodes(data=True) if ("friendly_city" in d['type'])]
        subdict = {k: v - 1000 for k, v in lengths.items() if k in subnodes}
        dist_array = sorted([(subnode_type[subnodes.index(k)], v) for (k,v) in subdict.items()], reverse=True)
        dist_dic = {i[0]:i[1] for i in dist_array}
        return dist_dic

    def find_optimal_fuel_distribution(self) -> list:
        if self.turn < 20:
            cities = []
            city_ids = []
        else: 
            # Creates a list of all demand nodes
            cities = self.cities_needing_fuel_index
            city_ids = self.cities_needing_fuel_id

        worker_pos = self.available_workers_positions
        # Creates a list of all the supply nodes
        workers = self.available_workers_index
        # Creates a list for the number of units of supply for each supply node
        supply =  self.available_workers_supply
        # Does worker supply contain special resources
        supply_special =  self.available_workers_special
        # Creates a list for the number of units of demand for each demand node
        demand = self.cities_needing_fuel_demand
        # City value based on the number of city tiles
        value = self.cities_needing_fuel_value
        # Worker distance to city in approximate move turns
        distance = self.worker_distance_to_cities * np.array([supply_special]).T
        # Creates the prob variable to contain the problem data
        prob = LpProblem("Fuel delivery problem",LpMaximize)
        # Creates a list of tuples containing all the possible routes for transport
        Routes = [(w,c) for w in workers for c in cities]
        # A dictionary called route_vars is created to contain the referenced variables (the routes)
        route_vars = LpVariable.dicts("Route",(workers,cities),0,None,LpInteger)
        route_y = LpVariable.dicts("Route_y",(workers,cities),0, cat="Binary")
        # The objective function is added to prob first
        prob += lpSum([route_vars[w][c] * (value[c]) for (w,c) in Routes])/10 - lpSum([distance[w][c] * route_y[w][c] for (w,c) in Routes])*4, "Sum of city value vs transporting cost"
        # The supply maximum constraints are added to prob for each supply node (worker)
        for w in workers:
            prob += lpSum([route_vars[w][c] for c in cities]) <= (supply[w]), "Sum of fuel out of worker %s"%w
            prob += lpSum([route_y[w][c] for c in cities]) <= 1, "workers y contraint %s"%w
            for c in cities:
                prob += route_vars[w][c] <= route_y[w][c] * 1000000, f"limit workers 1 {w} {c}"
                prob += route_vars[w][c] >= route_y[w][c], f"limit workers 2 {w} {c}"
        # The demand minimum constraints are added to prob for each demand node (city)
        for c in cities:
            prob += lpSum([route_vars[w][c] for w in workers]) <= demand[c], "Sum of fuel into city max city requirements %s"%c
        # Solve the optimization problem
        p = prob.solve(PULP_CBC_CMD(msg=0, timeLimit=2))
        orders = []
        for var in prob.variables():
            if "Route_y_" in var.name and var.value() == 1:
                orders.append(var.name[8:].split("_"))
        orders = [[worker_pos[int(i[0])], city_ids[int(i[1])]] for i in orders]

        logging.info(f"turn - {self.turn}, orders - {orders}, solution - {LpStatus[prob.status]}")
        if self.turn % 20 ==0:
            logging.info(f"city_ids - {city_ids}")
            logging.info(f"worker_pos - {worker_pos}")
            logging.info(f"supply - {supply}")
            logging.info(f"demand - {demand}")
            logging.info(f"value - {value}")
            logging.info(f"distance - {distance}, weight to city - {self.worker_distance_to_cities}")
            for var in prob.variables():
                logging.info(f"{var.name}: {var.value()}")
            for name, constraint in prob.constraints.items():
                logging.info(f"{name}: {constraint.value()}")
        return orders
