from utils.base_controller import BaseController
from lux.game import Game
from typing import Any
from lux.game_map import Cell
import numpy as np
import scipy.signal
import networkx as nx

class MapController(BaseController):
    def __init__(self, game_state: Game, observation: Any):
        super().__init__(game_state, observation)
        self.unit_locations = self.get_unit_locations()
        self.unit_locations_tuple = [(int(i.x), int(i.y)) for i in self.unit_locations]
        self.city_locations = self.get_city_locations()
        self.city_locations_tuple = [(int(i.x), int(i.y)) for i in self.city_locations]
        self.opponent_city_locations = self.get_opponent_unit_locations()
        self.opponent_city_locations_tuple = [(int(i.x), int(i.y)) for i in self.opponent_city_locations]
        self.opponent_unit_locations = self.get_opponent_city_locations()
        self.opponent_unit_locations_tuple = [(int(i.x), int(i.y)) for i in self.opponent_unit_locations]
        self.all_tiles = self.get_all_tiles()
        self.resource_tiles = self.all_tiles["resource_tiles"]
        self.wood_tiles = self.all_tiles["wood_tiles"]
        self.coal_tiles = self.all_tiles["coal_tiles"]
        self.uranium_tiles = self.all_tiles["uranium_tiles"]
        self.empty_tiles = self.all_tiles["empty_tiles"]
        self.map_adjacancies = self.get_map_adjacancies()
        self.city_adj = self.map_adjacancies["city_adj"]
        self.wood_adj = self.map_adjacancies["wood_adj"]
        self.coal_adj = self.map_adjacancies["coal_adj"]
        self.uranium_adj = self.map_adjacancies["uranium_adj"]
        self.best_build_spots = self.get_best_build_spots()
        self.graph_map = self.create_graph_map()
        

    def get_unit_locations(self) -> list:
        positions = []
        for unit in self.player.units:
            positions.append(unit.pos)
        return positions
    
    def get_city_locations(self) -> list:
        positions = []
        for city in self.player.cities.values():
            for citytile in city.citytiles:
                positions.append(citytile.pos)
        return positions
    
    def get_opponent_unit_locations(self) -> list:
        positions = []
        for unit in self.opponent.units:
            positions.append(unit.pos)
        return positions

    def get_opponent_city_locations(self) -> list:
        positions = []
        for city in self.opponent.cities.values():
            for citytile in city.citytiles:
                positions.append(citytile.pos)
        return positions

    def get_all_tiles(self) -> list[Cell]:
        resource_tiles = []
        empty_tiles = []
        wood_tiles = []
        coal_tiles = []
        uranium_tiles = []
        for y in range(self.height):
            for x in range(self.width):
                cell = self.map.get_cell(x, y)
                if cell.has_resource():
                    resource_tiles.append(cell)
                    if cell.resource.type.lower() == "wood":
                        wood_tiles.append((cell.pos.x, cell.pos.y))
                    elif cell.resource.type.lower() == "coal":
                        coal_tiles.append((cell.pos.x, cell.pos.y))
                    else:
                        uranium_tiles.append((cell.pos.x, cell.pos.y))
                else:
                    if cell.citytile is None:
                        empty_tiles.append((cell.pos.x, cell.pos.y))
        return {
            "resource_tiles": resource_tiles,
            "empty_tiles": empty_tiles,
            "wood_tiles": wood_tiles,
            "coal_tiles": coal_tiles,
            "uranium_tiles": uranium_tiles,
        }


    def get_map_adjacancies(self):
        city_matrix = np.zeros(shape=(self.width, self.height))
        city_resource_matrix = np.zeros(shape=(self.width, self.height))
        wood_matrix = np.zeros(shape=(self.width, self.height))
        coal_matrix = np.zeros(shape=(self.width, self.height))
        uranium_matrix = np.zeros(shape=(self.width, self.height))
        for y in range(self.height):
            for x in range(self.width):
                cell = self.map.get_cell(x, y)
                if cell.pos in self.city_locations:
                    city_matrix[x][y] = 15
                if cell.has_resource():
                    if cell.resource.type.lower() == "wood":
                        city_resource_matrix[x][y] = 3
                        if cell.resource.amount > 300:
                            wood_matrix[x][y] = 3
                        elif cell.resource.amount > 20:
                            wood_matrix[x][y] = 2
                if cell.has_resource():
                    if cell.resource.type.lower() == "coal" and self.player.researched_coal():
                        city_resource_matrix[x][y] = 2
                        coal_matrix[x][y] = 1
                if cell.has_resource():
                    if cell.resource.type.lower() == "uranium" and self.player.researched_uranium():
                        city_resource_matrix[x][y] = 1
                        uranium_matrix[x][y] = 1

        city_conv_matrix = np.array([[0,1,0], [1,0,1], [0,1,0]])
        city_resource_conv_matrix = np.array([[0,1,1,1,0], [1,1,1,1,1], [1,1,0,1,1], [1,1,1,1,1], [0,1,1,1,0]])
        resource_conv_matrix = np.array([[0,1,0], [1,1,1], [0,1,0]])
        city_adj1 = scipy.signal.convolve2d(city_matrix, city_conv_matrix, mode='same')
        city_adj2 = scipy.signal.convolve2d(city_resource_matrix, city_resource_conv_matrix, mode='same')
        city_adj = city_adj1 + city_adj2
        wood_adj = scipy.signal.convolve2d(wood_matrix, resource_conv_matrix, mode='same')
        coal_adj = scipy.signal.convolve2d(coal_matrix, resource_conv_matrix, mode='same') + wood_adj
        uranium_adj = scipy.signal.convolve2d(uranium_matrix, resource_conv_matrix, mode='same') + coal_adj + wood_adj

        return {
            "city_adj": city_adj,
            "wood_adj": wood_adj,
            "coal_adj": coal_adj,
            "uranium_adj": uranium_adj,
        }

    def get_best_build_spots(self) -> list[tuple]:
        city_tuples = []
        for y in range(self.height):
            for x in range(self.width):
                city_tuples.append((x, y, self.city_adj[x][y]))

        city_tuples2 = sorted(city_tuples, key=lambda tup: tup[2], reverse=True)
        return [(i[0], i[1]) for i in city_tuples2 if (i[0], i[1]) in self.empty_tiles][:round(self.height)]

    # create graph map of the game for optimisation
    def create_graph_map(self) -> nx.DiGraph:
        G = nx.grid_2d_graph(self.width, self.height)
        G2 = nx.DiGraph()
        for i in G.nodes:
            if i in self.opponent_city_locations_tuple:
                G2.add_node(i, type=["opponent_city"])
            elif i in self.city_locations_tuple:
                G2.add_node(i, type=["friendly_city"])
            elif self.wood_adj[i[0], i[1]] > 0:
                G2.add_node(i, type=["wood", "resource", "coal_researched", "uranium_researched"])
            elif self.coal_adj[i[0], i[1]] > 0:
                G2.add_node(i, type=["coal", "resource", "coal_researched", "uranium_researched"])
            elif self.uranium_adj[i[0], i[1]] > 0:
                G2.add_node(i, type=["uranium", "resource", "uranium_researched"])
            elif i in self.empty_tiles:
                G2.add_node(i, type=["empty"])
            else:
                G2.add_node(i, type=["unkown"])

        G2.add_edges_from(G.edges)
        for s, t in G2.edges():
            G2.add_edge(t,s)

        for s, t in G2.edges():
            if t in self.opponent_city_locations_tuple:
                weight=1000
                weight_ac=1000
            elif t in self.unit_locations_tuple and t not in self.city_locations_tuple:
                weight=1000
                weight_ac=1000
            elif t in self.city_locations_tuple:
                weight=1
                weight_ac=1000
            elif t in self.opponent_unit_locations_tuple:
                weight=12
                weight_ac=12
            else:
                weight=3
                weight_ac=3
            G2[s][t]['weight'] = weight
            G2[s][t]['weight_ac'] = weight_ac
        return G2
    
    def add_weight(self, t: tuple, avoid_citys=False) -> None:
        s_list = [(t[0]+1, t[1]), (t[0]-1, t[1]), (t[0], t[1]+1), (t[0], t[1]-1)]
        s_list = [i for i in s_list if self.width > i[0] >= 0 and self.height > i[1] >= 0]
        for s in s_list:
            if avoid_citys:
                self.graph_map[s][t]['weight'] = 3
                self.graph_map[s][t]['weight_ac'] = 1000
            else:
                self.graph_map[s][t]['weight'] = 1000
                self.graph_map[s][t]['weight_ac'] = 1000
    
    def remove_weight(self, t: tuple, avoid_citys=False) -> None:
        s_list = [(t[0]+1, t[1]), (t[0]-1, t[1]), (t[0], t[1]+1), (t[0], t[1]-1)]
        s_list = [i for i in s_list if self.width > i[0] >= 0 and self.height > i[1] >= 0]
        for s in s_list:
            if avoid_citys:
                self.graph_map[s][t]['weight'] = 3
                self.graph_map[s][t]['weight_ac'] = 1000
            else:
                self.graph_map[s][t]['weight'] = 3
                self.graph_map[s][t]['weight_ac'] = 1
