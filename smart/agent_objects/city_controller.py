import numpy as np
from pulp.constants import LpMinimize
from utils.base_controller import BaseController
from agent_objects.map_controller import MapController
from pulp.pulp import LpProblem, LpVariable, lpSum, PULP_CBC_CMD
from pulp.constants import LpMaximize, LpInteger, LpStatus
from lux.game import Game
from lux.game_objects import Unit
from typing import Any, List, Dict
import networkx as nx
import logging

class CityController(BaseController):
    def __init__(self, game_state: Game, observation: Any, mc: MapController):
        super().__init__(game_state, observation)
        self.mc = mc
        self.available_workers = self.get_available_workers()
        self.available_workers_positions = self.available_workers.get("positions")
        self.available_workers_supply = self.available_workers.get("supply")
        self.available_workers_special = self.available_workers.get("supply_special")
        self.available_workers_index = self.available_workers.get("index")
        self.available_workers_unit = self.available_workers.get("unit")
        self.cities_needing_fuel = self.get_cities_needing_fuel()
        self.cities_needing_fuel_id = self.cities_needing_fuel.get("city_id")
        self.cities_needing_fuel_demand = self.cities_needing_fuel.get("demand")
        self.cities_needing_fuel_value = self.cities_needing_fuel.get("value")
        self.cities_needing_fuel_index = self.cities_needing_fuel.get("index")
        self.cities_needing_fuel_turns = self.cities_needing_fuel.get("fuel_turns")
        self.citytiles = self.get_city_tiles()
        self.fuel_distribution_orders = self.find_optimal_fuel_distribution()
        self.workers_on_fuel = [i[0] for i in self.fuel_distribution_orders]
        self.wood_workers = self.get_wood_workers()
        self.wood_workers_pos = self.wood_workers.get("wood_workers_pos")
        self.wood_workers_unit = self.wood_workers.get("wood_workers")
        self.wood_exploit_orders = self.find_optimal_wood_assignment()

    def use_cities(self, actions):
        new_workers = 0
        new_research = 0
        # iterate over cities and do something
        for citytile in self.citytiles:
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
        worker = []
        for unit in self.player.units:
            if unit.cargo.wood + (unit.cargo.coal*4) + (unit.cargo.uranium*8) >= 50 and (int(unit.id[2:])%3!=0 or not self.mc.dist_wood_clusters):
                positions.append((unit.pos.x, unit.pos.y))
                supply.append(unit.cargo.wood + (unit.cargo.coal*10) + (unit.cargo.uranium*40))
                worker.append(unit)
                if (unit.cargo.coal*10) + (unit.cargo.uranium*40) > 30:
                    supply_special.append(0.1)
                else:
                    supply_special.append(1)
        
        return {
            "positions": [positions[i] for i in np.argsort(supply)[::-1]][:35],
            "supply": [supply[i] for i in np.argsort(supply)[::-1]][:35],
            "supply_special": [supply_special[i] for i in np.argsort(supply)[::-1]][:35],
            "index": [i for i in range(len(supply))][:35],
            "unit": [worker[i] for i in np.argsort(supply)[::-1]][:35],
            }
    
    def get_wood_workers(self) -> Dict[str, list]:
        wood_workers_pos: List[tuple] = []
        wood_workers: List[Unit] = []
        for i in self.player.units:
            if int(i.id[2:])%3==0:
                total_cargo = i.cargo.wood+i.cargo.coal+i.cargo.uranium
                if i.is_worker() and (i.pos.x, i.pos.y) not in self.workers_on_fuel and total_cargo < 80:
                    wood_workers_pos += [(i.pos.x, i.pos.y)]
                    wood_workers += [i]
        return {
            "wood_workers_pos": wood_workers_pos,
            "wood_workers": wood_workers,
            }

    def get_city_tiles(self):
        city_tiles = []
        fuel_turns = []
        for city in self.player.cities.values():
            ft = int(city.fuel/city.get_light_upkeep())
            for citytile in city.citytiles:
                nearby_resources = self.mc.nearby_resources[citytile.pos.x][citytile.pos.y]
                if nearby_resources < 500:
                    ft = 99
                city_tiles.append(citytile)
                fuel_turns.append(ft)
        return [x for _, x in sorted(zip(fuel_turns, city_tiles), key=lambda pair: pair[0])]

    def get_cities_needing_fuel(self) -> dict:
        city_id = []
        demand = []
        fuel_turns = []
        value = []
        index = []
        for city in self.player.cities.values():
            fuel = city.fuel
            upkeep = city.get_light_upkeep()
            nights_remaining = ((360 - self.turn)//40)*10 + min(((360 - self.turn)%40),10)
            if fuel < (upkeep * nights_remaining) + 300:
                city_id.append(f"{city.cityid}")
                demand.append(max((((upkeep * nights_remaining) - fuel + 300)), 100))
                value.append(max(min(len(city.citytiles)/4, 4),1))
                index.append((max(index)+1 if index else 0))
                fuel_turns.append((((int(fuel/upkeep)//10) * 40)  - (30 - self.turns_until_night) + (int(fuel/upkeep)%10)) + 30)
            
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
        wu = self.available_workers_unit
        dist_matrix = np.ones(shape=(no_w, no_c))
        for w in range(len(wp)):
            worker: Unit = wu[w]
            cargo_fuel_value = max((worker.cargo.wood - 50)/4, 0) + max(worker.cargo.coal - 13, 0) + max(worker.cargo.uranium - 8, 0)
            fuel_turns = int(cargo_fuel_value) + self.turns_until_night + (30 if max((cargo_fuel_value - 10),0) > 0 else 0)
            dist_dic = self.find_distance_to_type(source=wp[w], type="friendly_city", weight="weight_ac", adj=999)
            for c in dist_dic.items():
                c_name = c[0]
                c_dist = c[1]
                cid = self.cities_needing_fuel_id.index(c_name) if c_name in self.cities_needing_fuel_id else "No match"
                if cid != "No match":
                    adjusted_distance = c_dist + ((((c_dist - self.turns_until_night)//40)*10 + min((c_dist - self.turns_until_night)%40,10))*2) + worker.cooldown
                    if adjusted_distance <= self.cities_needing_fuel_turns[cid] and adjusted_distance <= fuel_turns:
                        dist_matrix[w][cid] = adjusted_distance
                    else:
                        dist_matrix[w][cid] = 99999
        return dist_matrix

    def get_worker_distance_to_wood(self) -> np.ndarray:
        if self.mc.dist_wood_clusters:
            unique_clusters = [i for i in set(self.mc.dist_wood_clusters)]
            wu = self.wood_workers_unit
            wp = self.wood_workers_pos
            no_w = len(wu)
            no_c = len(unique_clusters)
            dist_matrix = np.ones(shape=(no_w, no_c))
            for w in range(len(wu)):
                worker: Unit = wu[w]
                total_cargo = worker.cargo.wood + worker.cargo.coal + worker.cargo.uranium
                fuel_turns = int(total_cargo/4) + self.turns_until_night + (30 if max(((total_cargo/4) - 10),0) > 0 else 0)
                dist_dic = self.find_distance_to_type(source=wp[w], type="dist_wood", weight="weight")
                for c in dist_dic.items():
                    c_name = c[0]
                    c_dist = c[1]
                    cid = unique_clusters.index(c_name) if c_name in unique_clusters else "No match"
                    if cid != "No match":
                        adjusted_distance = c_dist + ((((c_dist - self.turns_until_night)//40)*10 + min((c_dist - self.turns_until_night)%40,10))*2) + worker.cooldown
                        dist_matrix[w][cid] = adjusted_distance
                        if adjusted_distance <= fuel_turns:
                            dist_matrix[w][cid] = adjusted_distance
                        else:
                            dist_matrix[w][cid] = 99999
            return dist_matrix

        # find closest route to node type
    def find_distance_to_type(self, source, type, weight="weight", adj=0) -> dict:
        G = self.mc.graph_map
        #Calculate the length of paths from source to all other nodes
        lengths=nx.single_source_dijkstra_path_length(G, source, weight=weight)
        #We are only interested in a particular type of node
        subnodes = [name for name, d in G.nodes(data=True) if (type in d['type'])]
        subnode_type = [d['type'][-1] for name, d in G.nodes(data=True) if (type in d['type'])]
        subdict = {k: v - adj for k, v in lengths.items() if k in subnodes}
        dist_array = sorted([(subnode_type[subnodes.index(k)], v) for (k,v) in subdict.items()], reverse=True)
        dist_dic = {i[0]:i[1] for i in dist_array}
        return dist_dic

    def find_optimal_fuel_distribution(self) -> list:
        cities = self.cities_needing_fuel_index
        city_ids = self.cities_needing_fuel_id
        if self.turn < 25 or len(cities) == 0:
            logging.info("cities don't need fuel")
            return [[None],[None]]
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
        city_tiles = self.cities_needing_fuel_value
        # Turns cities can survive for
        turns = self.cities_needing_fuel_turns
        # City value based on the number of city tiles and the number of turns it can survive
        value = [city_tiles[i]/max(np.log10(max(turns[i],1))**3.5, 0.5) for i in range(len(turns))]
        # Worker distance to city in approximate move turns
        distance = self.get_worker_distance_to_cities()
        distance = np.clip(distance, 5, 99999) * np.array([supply_special]).T
        # Creates the prob variable to contain the problem data
        prob = LpProblem("fuel-delivery-problem",LpMaximize)
        # Creates a list of tuples containing all the possible routes for transport
        Routes = [(w,c) for w in workers for c in cities]
        # A dictionary called route_vars is created to contain the referenced variables (the routes)
        route_vars = LpVariable.dicts("Route",(workers,cities),0,None,LpInteger)
        route_y = LpVariable.dicts("Route_y",(workers,cities),0, cat="Binary")
        # The objective function is added to prob first
        prob += lpSum([route_vars[w][c] * (value[c]) for (w,c) in Routes])/10 - lpSum([distance[w][c] * route_y[w][c] for (w,c) in Routes])*2, "Sum of city value vs transporting cost"
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
        p = prob.solve(PULP_CBC_CMD(msg=0, timeLimit=.5))
        orders = []
        for var in prob.variables():
            if "Route_y_" in var.name and var.value() == 1:
                orders.append(var.name[8:].split("_"))
        orders = [[worker_pos[int(i[0])], city_ids[int(i[1])]] for i in orders]

        logging.info(f"turn - {self.turn}, orders - {orders}, solution - {LpStatus[prob.status]}")
        if self.turn % 20 ==0:
            for var in prob.variables():
                logging.info(f"{var.name}: {var.value()}")
        return orders


    def find_optimal_wood_assignment(self) -> list:
        if not self.mc.dist_wood_clusters:
            logging.info("no wood clusters left")
            return [[None],[None]]
        else:
            unique_clusters = set(self.mc.dist_wood_clusters)
            # Creates a list of all demand nodes
            clusters = [i for i in range(len(unique_clusters))]
            wood_cluster_names = [i for i in unique_clusters]
            worker_pos = self.wood_workers_pos
            # Creates a list of all the supply nodes
            workers = [i for i in range(len(worker_pos))]
            # Worker distance to city in approximate move turns
            distance = self.get_worker_distance_to_wood()
            # Creates the prob variable to contain the problem data
            prob = LpProblem("wood-exploitation-problem",LpMaximize)
            # Creates a list of tuples containing all the possible routes for transport
            Routes = [(w,c) for w in workers for c in clusters]
            # A dictionary called route_vars is created to contain the referenced variables (the routes)
            route_y = LpVariable.dicts("Route_y",(workers,clusters),0, cat="Binary")
            # The objective function is added to prob first
            prob += lpSum([route_y[w][c] for (w,c) in Routes])*100 - lpSum([distance[w][c] * route_y[w][c] for (w,c) in Routes]), "Sum new wood value vs worker distance"
            # The supply maximum constraints are added to prob for each supply node (worker)
            for w in workers:
                prob += lpSum([route_y[w][c] for c in clusters]) <= 1, "Max 1 job per worker %s"%w
            # The demand minimum constraints are added to prob for each demand node (city)
            for c in clusters:
                prob += lpSum([route_y[w][c] for w in workers]) <= 1, "Max 1 worker sent to each cluster %s"%c

            # Solve the optimization problem
            p = prob.solve(PULP_CBC_CMD(msg=0, timeLimit=.4))

            orders = []
            for var in prob.variables():
                if "Route_y_" in var.name and var.value() == 1:
                    orders.append(var.name[8:].split("_"))
            orders = [[worker_pos[int(i[0])], wood_cluster_names[int(i[1])]] for i in orders]
            return orders