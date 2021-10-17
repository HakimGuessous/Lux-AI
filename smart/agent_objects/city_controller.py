from utils.base_controller import BaseController


class CityController(BaseController):

    def use_cities(self, actions):
        new_workers = 0
        new_research = 0
        # iterate over cities and do something
        for k, city in self.player.cities.items():
            id = city.cityid
            fuel = city.fuel
            fuel_burn = city.get_light_upkeep()

            for citytile in city.citytiles:
                if citytile.can_act():
                    if (self.no_workers + new_workers) < self.no_city_tiles:
                        actions.append(citytile.build_worker())
                        new_workers += 1
                    
                    elif (self.player.research_points + new_research) < 200:
                        actions.append(citytile.research())
                        new_research += 1