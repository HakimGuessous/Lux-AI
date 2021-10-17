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
    def __init__(self, game_state: Game, observation: Any, orders: list, mc: MapController):
        super().__init__(game_state, observation)
        self.mc = mc
        self.unit_positions = self.mc.unit_locations
        self.fuel_distribution_orders = orders
        self.fuel_distribution_workers = [i[0] for i in orders]

    def use_units(self, actions: list):
        for unit in self.player.units:
            self.unit = unit
            self.use_unit(actions)

    def get_closest_resource(self, actions: list, weight="weight") -> str:
        unit_position_cood = (self.unit.pos.x, self.unit.pos.y)
        if self.player.researched_uranium():
            closest_resource, length = self.find_closest_type("uranium_researched", unit_position_cood, weight, self.mc.uranium_adj)
        elif self.player.researched_coal():
            closest_resource, length = self.find_closest_type("coal_researched", unit_position_cood, weight, self.mc.coal_adj)
        else:
            closest_resource, length = self.find_closest_type("wood", unit_position_cood, weight, self.mc.wood_adj)
        if closest_resource and length <500:
            route_tuple = (closest_resource[0] if len(closest_resource)==1 else closest_resource[1])
            return self.move_unit_along_path(route_tuple, actions)

    def use_unit(self, actions: list) -> Any:
        unit = self.unit
        unit_pos_tuple = (unit.pos.x, unit.pos.y)
        if unit.is_worker() and unit.can_act():
            if unit_pos_tuple in self.fuel_distribution_workers:
                worker_order = [i for i in self.fuel_distribution_orders if i[0] == unit_pos_tuple][0]
                closest_city_tile, length = self.find_closest_type(worker_order[1], unit_pos_tuple, weight="weight_ac")
                route_tuple = closest_city_tile[1]
                return self.move_unit_along_path(route_tuple, actions)

            elif 100 > unit.get_cargo_space_left() > 0:
                return self.get_closest_resource(actions, weight="weight_ac")
            elif unit.get_cargo_space_left() > 0:
                return self.get_closest_resource(actions)
            else:
                self.build_on_best_spot(actions)
                        

    def move_randomly(self, source: tuple) -> tuple:
        x = max(min(self.width-1, source[0] + random.randint(-1,1)), 0)
        y = max(min(self.width-1, source[1]+ random.randint(-1,1)), 0)
        return (x, y)
    
    def find_fastest_route(self, source: tuple, target: tuple, avoid_cities: bool=False, weight="weight", max_length=500) -> tuple:
        # # otherwise move to the fastest way to target
        G = self.mc.graph_map
        if not avoid_cities:
            path = nx.dijkstra_path(G, source, target, weight="weight")
            length = nx.shortest_path_length(G, source, target, weight="weight")
        else:
            path = nx.dijkstra_path(G, source, target, weight="weight_ac")
            length = nx.shortest_path_length(G, source, target, weight="weight_ac")

        if len(path)>1 and length < max_length:
            path_out = path[1]
        else:
            path_out = self.move_randomly(source)
        return path_out
    
    # find closest route to node type
    def find_closest_type(self, typeofnode, source, weight, matrix=False):
        G = self.mc.graph_map
        #Calculate the length of paths from source to all other nodes
        lengths=nx.single_source_dijkstra_path_length(G, source, weight=weight)
        paths = nx.single_source_dijkstra_path(G, source, weight=weight)
        #We are only interested in a particular type of node
        subnodes = [name for name, d in G.nodes(data=True) if (typeofnode in d['type'])]
        subdict = {k: v for k, v in lengths.items() if k in subnodes}
        if isinstance(matrix, np.ndarray):
            for i, j in subdict.items():
                subdict[i] = j - matrix[i[0], i[1]]
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
        route = self.find_closest_type(typeofnode="empty", source=unit_position_cood, weight="weight_ac", matrix=self.mc.city_adj)
        if route[1] is not None and route[1] < 500:
            if len(route[0]) > 1:
                route_tuple = route[0][1]
            else:
                route_tuple = route[0][0]
            build_spot = self.map.get_cell(route_tuple[0], route_tuple[1])

            if self.unit.pos.equals(build_spot.pos):
                actions.append(self.unit.build_city())
            else:
                return self.move_unit_along_path(route_tuple, actions)
