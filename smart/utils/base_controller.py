from typing import Any
from lux.game import Game


class BaseController:
        def __init__(self, game_state: Game, observation: Any):
            self.player = game_state.players[observation.player]
            self.opponent = game_state.players[(observation.player + 1) % 2]
            self.width, self.height = game_state.map.width, game_state.map.height
            self.no_workers = len(self.player.units)
            self.no_city_tiles = sum(len(v.citytiles) for v in self.player.cities.values())
            self.map = game_state.map
            self.turn = game_state.turn
            self.turns_until_night = max(30 - (game_state.turn%40),0)