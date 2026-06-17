import torch


class MechanismStateBuilder:
    """
    Builds the state observed by the upper-level
    mechanism policy.

    Output structure:

        [ spot_price
          avg_price
          avg_waste_price
          avg_inventory
          avg_waste_inventory ]

    Dimension:

        5 * num_commodities
    """

    def __init__(self, num_commodities):
        self.num_commodities = num_commodities

    def build_from_simulator(self, simulator):

        spot_price = simulator.state.spot_price

        avg_price = simulator.state.price.mean(dim=0)

        avg_waste_price = (
            simulator.state.waste_price.mean(dim=0)
        )

        avg_inventory = (
            simulator.state.inv.mean(dim=0)
        )

        avg_waste_inventory = (
            simulator.state.waste_inv.mean(dim=0)
        )

        state = torch.cat(
            [
                spot_price,
                avg_price,
                avg_waste_price,
                avg_inventory,
                avg_waste_inventory,
            ],
            dim=0,
        )

        return state

    def build_from_tensors(
        self,
        spot_price,
        price,
        waste_price,
        inventory,
        waste_inventory,
    ):

        state = torch.cat(
            [
                spot_price,
                price.mean(dim=0),
                waste_price.mean(dim=0),
                inventory.mean(dim=0),
                waste_inventory.mean(dim=0),
            ],
            dim=0,
        )

        return state

    @property
    def state_dim(self):
        return 5 * self.num_commodities
