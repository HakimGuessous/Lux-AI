from typing import overload
from lux.game import Game
from lux.constants import Constants

from agent_objects.unit_controller import UnitControler
from agent_objects.city_controller import CityController

DIRECTIONS = Constants.DIRECTIONS
game_state = None


def agent(observation, configuration):
    global game_state

    ### Do not edit ###
    if observation["step"] == 0:
        game_state = Game()
        game_state._initialize(observation["updates"])
        game_state._update(observation["updates"][2:])
        game_state.id = observation.player
    else:
        game_state._update(observation["updates"])
    
    actions = []

    ### AI Code goes down here! ### 
    player = game_state.players[observation.player]
    opponent = game_state.players[(observation.player + 1) % 2]

    # we iterate over all our units and do something with them
    
    uc = UnitControler(game_state, observation)
    uc.use_units(actions)

    # iterate over cities and do something
    cc = CityController(game_state, observation)
    cc.use_cities(actions)

    
    return actions
