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
        self.settle_value = self.map_adjacancies["settle_value"]
        self.wood_adj = self.map_adjacancies["wood_adj"]
        self.coal_adj = self.map_adjacancies["coal_adj"]
        self.uranium_adj = self.map_adjacancies["uranium_adj"]
        self.nearby_resources = self.map_adjacancies["nearby_resources"]
        self.wood_dist = self.map_adjacancies["wood_dist"]
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
        city_matrix2 = np.zeros(shape=(self.width, self.height))
        city_resource_matrix = np.zeros(shape=(self.width, self.height))
        city_resource_matrix2 = np.zeros(shape=(self.width, self.height))
        wood_matrix = np.zeros(shape=(self.width, self.height))
        wood_dist_matrix = np.zeros(shape=(self.width, self.height))
        coal_matrix = np.zeros(shape=(self.width, self.height))
        uranium_matrix = np.zeros(shape=(self.width, self.height))
        for y in range(self.height):
            for x in range(self.width):
                cell = self.map.get_cell(x, y)
                if cell.pos in self.city_locations:
                    city_matrix[x][y] = 1
                    city_matrix2[x][y] = 4
                    wood_dist_matrix[x][y] += -40
                elif cell.pos in self.opponent_city_locations:
                    wood_dist_matrix[x][y] += -10
                elif cell.pos in self.opponent_unit_locations:
                    wood_dist_matrix[x][y] += -6
                elif cell.has_resource():
                    if cell.resource.type.lower() == "wood":
                        city_resource_matrix[x][y] = 1.5
                        wood_matrix[x][y] = 2
                        wood_dist_matrix[x][y] = 2
                        city_resource_matrix2[x][y] = cell.resource.amount
                    elif cell.resource.type.lower() == "coal":
                        coal_matrix[x][y] = 12
                        if self.player.researched_coal():
                            city_resource_matrix[x][y] = 2
                            city_resource_matrix2[x][y] = cell.resource.amount * 10

                    elif cell.resource.type.lower() == "uranium":
                        uranium_matrix[x][y] = 12
                        if self.player.researched_uranium():
                            city_resource_matrix[x][y] = 2
                            city_resource_matrix2[x][y] = cell.resource.amount * 40

        city_conv_matrix = np.array([[0,1,0], [1,0,1], [0,1,0]])
        city_resource_conv_matrix = np.array([[0,1,1,1,0], [1,1,1,1,1], [1,1,0,1,1], [1,1,1,1,1], [0,1,1,1,0]])
        big_conv_matrix = np.array([[0,0,0,0,0,1,0,0,0,0,0],
                                    [0,0,0,0,1,1,1,0,0,0,0],
                                    [0,0,0,1,1,1,1,1,0,0,0],
                                    [0,0,1,1,1,1,1,1,1,0,0],
                                    [0,1,1,1,1,1,1,1,1,1,0],
                                    [1,1,1,1,1,1,1,1,1,1,1],
                                    [0,1,1,1,1,1,1,1,1,1,0],
                                    [0,0,1,1,1,1,1,1,1,0,0],
                                    [0,0,0,1,1,1,1,1,0,0,0],
                                    [0,0,0,0,1,1,1,0,0,0,0],
                                    [0,0,0,0,0,1,0,0,0,0,0]])

        resource_conv_matrix = np.array([[0,1,0], [1,1,1], [0,1,0]])
        city_adj1 = scipy.signal.convolve2d(city_matrix, city_conv_matrix, mode='same')
        city_adj2 = scipy.signal.convolve2d(city_matrix2, city_conv_matrix, mode='same')
        city_adj3 = scipy.signal.convolve2d(city_resource_matrix, city_resource_conv_matrix, mode='same')
        settle_value = city_adj2 + city_adj3
        wood_adj = scipy.signal.convolve2d(wood_matrix, resource_conv_matrix, mode='same')
        coal_adj = scipy.signal.convolve2d(coal_matrix, resource_conv_matrix, mode='same')
        uranium_adj = scipy.signal.convolve2d(uranium_matrix, resource_conv_matrix, mode='same') + coal_adj
        wood_dist = scipy.signal.convolve2d(wood_dist_matrix, big_conv_matrix, mode='same')
        for y in range(self.height):
            for x in range(self.width):
                cell = self.map.get_cell(x, y)
                if wood_adj[x][y] != 0 and cell.has_resource() == False:
                    wood_adj[x][y] += 3

        nearby_resources = scipy.signal.convolve2d(city_resource_matrix2, big_conv_matrix, mode='same')

        return {
            "city_adj": city_adj1,
            "settle_value": settle_value,
            "wood_adj": wood_adj,
            "coal_adj": coal_adj,
            "uranium_adj": uranium_adj,
            "nearby_resources": nearby_resources,
            "wood_dist": wood_dist,
        }

    # create graph map of the game for optimisation
    def create_graph_map(self) -> nx.DiGraph:
        G = nx.grid_2d_graph(self.width, self.height)
        G2 = nx.DiGraph()
        for i in G.nodes:
            type = []
            if i in self.opponent_city_locations_tuple:
                G2.add_node(i, type=["opponent_city"])
            elif i in self.city_locations_tuple:
                city = self.map.get_cell(i[0], i[1])
                G2.add_node(i, type=["friendly_city", f"{city.citytile.cityid}"])
            else:
                if self.wood_adj[i[0], i[1]] > 0:
                    type += ["wood", "resource", "coal_researched", "uranium_researched"]
                if self.coal_adj[i[0], i[1]] > 0:
                    type += ["coal", "resource", "coal_researched", "uranium_researched", "special"]
                if self.uranium_adj[i[0], i[1]] > 0:
                    type += ["uranium", "resource", "uranium_researched", "special"]
                if i in self.empty_tiles and (self.nearby_resources[i[0]][i[1]] > 2000 or self.city_adj[i[0]][i[1]] >=2):
                    type += ["settle"]
                elif i in self.empty_tiles:
                    type += ["empty"]
                else:
                    type += ["unkown"]
                G2.add_node(i, type=type)

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
                weight=6
                weight_ac=6
            else:
                weight=2
                weight_ac=2
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
