import math
from typing import Any, List
from lux.game_map import Cell
from lux.constants import Constants
from lux.game_objects import Unit, CityTile
from lux.game_map import Cell
from lux.game import Game
import random
import networkx as nx
import numpy as np
from utils.base_controller import BaseController
from agent_objects.map_controller import MapController

class UnitControler(BaseController):
    def __init__(self, game_state: Game, observation: Any, fuel_orders: list, wood_orders: list, mc: MapController):
        super().__init__(game_state, observation)
        self.mc = mc
        self.unit_positions = self.mc.unit_locations
        self.fuel_distribution_orders = fuel_orders
        self.fuel_distribution_workers = [i[0] for i in fuel_orders]
        self.wood_exploit_orders = wood_orders
        self.wood_exploit_workers = [i[0] for i in wood_orders]

    def use_units(self, actions: list):
        for unit in self.player.units:
            self.unit = unit
            self.use_unit(actions)

    def get_closest_resource(self, actions: list, method="closest", weight="weight") -> str:
        unit_position_cood = (self.unit.pos.x, self.unit.pos.y)
        if method=="closest":
            if self.player.researched_uranium():
                resource, length = self.find_closest_type("uranium_researched", unit_position_cood, weight, self.mc.uranium_adj)
            elif self.player.researched_coal():
                resource, length = self.find_closest_type("coal_researched", unit_position_cood, weight, self.mc.coal_adj)
            else:
                resource, length = self.find_closest_type("wood", unit_position_cood, weight, self.mc.wood_adj)
        elif method=="special":
            if self.player.research_points > 185:
                resource, length = self.find_closest_type("special", unit_position_cood, weight, self.mc.uranium_adj)
            elif self.player.research_points > 25:
                resource, length = self.find_closest_type("coal", unit_position_cood, weight, self.mc.coal_adj)
            else:
                resource=None
                length=None
        if resource and length <500:
            route_tuple = (resource[0] if len(resource)==1 else resource[1])
            return self.move_unit_along_path(route_tuple, actions)
        else: 
            return self.move_to_closest_city(actions, unit_position_cood)

    def use_unit(self, actions: list) -> Any:
        unit = self.unit
        cur_cell = self.map.get_cell_by_pos(self.unit.pos)
        id = int(self.unit.id[2:])
        unit_pos_tuple = (unit.pos.x, unit.pos.y)
        if unit.is_worker() and unit.can_act():
            on_resource = (self.mc.wood_adj[unit.pos.x][unit.pos.y] > 0 \
            or (self.mc.coal_adj[unit.pos.x][unit.pos.y] and self.player.researched_coal()) \
            or (self.mc.uranium_adj[unit.pos.x][unit.pos.y] and self.player.researched_uranium())) \
            and (unit.pos.x, unit.pos.y) not in self.mc.city_locations_tuple
            if unit_pos_tuple in self.fuel_distribution_workers:
                worker_order = [i for i in self.fuel_distribution_orders if i[0] == unit_pos_tuple][0]
                closest_city_tile, length = self.find_closest_type(worker_order[1], unit_pos_tuple, weight="weight_ac")
                route_tuple = closest_city_tile[1]
                return self.move_unit_along_path(route_tuple, actions)

            elif unit.get_cargo_space_left() <= 20 and cur_cell.has_resource() and cur_cell.resource.type=="wood" and cur_cell.resource.amount >= 60:
                if not self.build_on_best_spot(actions):
                    if on_resource:
                        return None
                    return self.get_closest_resource(actions)

            elif unit.get_cargo_space_left() == 0:
                if not self.build_on_best_spot(actions):
                    if on_resource:
                        return None
                    return self.get_closest_resource(actions)

            elif unit_pos_tuple in self.wood_exploit_workers:
                worker_order = [i for i in self.wood_exploit_orders if i[0] == unit_pos_tuple][0]
                closest_cluster_tile, length = self.find_closest_type(worker_order[1], unit_pos_tuple, weight="weight")
                if closest_cluster_tile:
                    route_tuple = (closest_cluster_tile[1] if len(closest_cluster_tile) > 1 else closest_cluster_tile[0])
                    return self.move_unit_along_path(route_tuple, actions)

            elif id%4 == 0 and self.player.research_points > 25 and (id%3!=0 or self.mc.dist_wood_clusters):
                if on_resource:
                        return None
                return self.get_closest_resource(actions, method="special")

            else:
                if on_resource:
                        return None
                return self.get_closest_resource(actions)


    def move_randomly(self, actions, source: tuple) -> tuple:
        x = max(min(self.width-1, source[0] + random.randint(-1,1)), 0)
        y = max(min(self.width-1, source[1]+ random.randint(-1,1)), 0)
        route_cell = self.map.get_cell(x, y)
        direction = self.unit.pos.direction_to(route_cell.pos)
        return actions.append(self.unit.move(direction))

    def move_to_closest_city(self, actions, source: tuple):
        closest_city_tile, length = self.find_closest_type("friendly_city", source, weight="weight")
        route_tuple = (closest_city_tile[1] if len(closest_city_tile)>1 else closest_city_tile[0])
        return self.move_unit_along_path(route_tuple, actions)
    
    # find closest route to node type
    def find_closest_type(self, typeofnode, source, weight, matrix=False, multiplier=1):
        G = self.mc.graph_map
        #Calculate the length of paths from source to all other nodes
        lengths=nx.single_source_dijkstra_path_length(G, source, weight=weight)
        paths = nx.single_source_dijkstra_path(G, source, weight=weight)
        #We are only interested in a particular type of node
        subnodes = [name for name, d in G.nodes(data=True) if (typeofnode in d['type'])]
        subdict = {k: v for k, v in lengths.items() if k in subnodes}
        cargo_turns = (self.unit.cargo.wood/4)+self.unit.cargo.coal+self.unit.cargo.uranium >= 10
        if not isinstance(matrix, np.ndarray):
            matrix = np.zeros((self.width, self.height))
        for i, j in subdict.items():
            if j + self.unit.cooldown < self.turns_until_night:
                subdict[i] = j*multiplier - matrix[i[0], i[1]] + self.unit.cooldown
                lengths[i] = j*multiplier - matrix[i[0], i[1]] + self.unit.cooldown
            elif cargo_turns and typeofnode != "settle":
                subdict[i] =  (j+(j-self.turns_until_night))*multiplier - matrix[i[0], i[1]] + self.unit.cooldown
                lengths[i] = (j+(j-self.turns_until_night))*multiplier - matrix[i[0], i[1]] + self.unit.cooldown
            else:
                subdict[i] = 99999
                lengths[i] = 99999
        #return the smallest of all lengths to get to typeofnode
        if subdict: 
            nearest =  min(subdict, key=subdict.get)
            return paths[nearest], lengths[nearest]
        else: #not found, no path from source to typeofnode
            return None, None
    
    def move_unit_along_path(self, path: tuple, actions: list):
        unit_position_cood = (self.unit.pos.x, self.unit.pos.y)
        route_cell = self.map.get_cell(path[0], path[1])
        direction = self.unit.pos.direction_to(route_cell.pos)
        current_cell = self.map.get_cell(unit_position_cood[0], unit_position_cood[1])
        if not route_cell.pos.equals(current_cell.pos):
            if current_cell.citytile is None:
                self.mc.add_weight(path)
                self.mc.remove_weight(unit_position_cood)
            else:
                self.mc.add_weight(path)
                self.mc.remove_weight(unit_position_cood, avoid_citys=True)

        return actions.append(self.unit.move(direction))


    def build_on_best_spot(self, actions: list):
        unit_position_cood = (self.unit.pos.x, self.unit.pos.y)
        multiplier = 1
        if self.turn < 60:
            multiplier = 3
        route = self.find_closest_type(typeofnode="settle", source=unit_position_cood, weight="weight_ac", matrix=self.mc.settle_value,multiplier=multiplier)
        if route[1] is not None and route[1] < 500:
            if len(route[0]) > 1:
                route_tuple = route[0][1]
                self.move_unit_along_path(route_tuple, actions)
                return True
            else:
                if self.turns_until_night > 1:
                    actions.append(self.unit.build_city())
                return True
        return None
        
