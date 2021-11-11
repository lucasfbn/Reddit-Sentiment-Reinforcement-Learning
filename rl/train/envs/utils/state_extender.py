import pandas as pd


class StateExtender:

    @staticmethod
    def get_new_shape_state(shape):
        """
        Returns the new state shape BEFORE actually applying an extender method. This is used to determine state shape
        before starting the training.

        Returns: New shape
        """
        raise NotImplementedError

    @staticmethod
    def add_inventory_state(state: pd.DataFrame, inventory_state: int):
        """
        Adds information to the state whether the agent currently has a stock in the inventory or not. (E.g. bought a
        stock in the past and didn't sell it yet)

        Returns: Extended state
        """
        raise NotImplementedError


class StateExtenderNN(StateExtender):
    pass


class StateExtenderCNN(StateExtender):

    @staticmethod
    def get_new_shape_state(shape):
        new_shape = (shape[0] + 1, shape[1])
        return new_shape

    @staticmethod
    def add_inventory_state(state, inventory_state):
        state_columns = list(state.columns)
        inventory_state_list = [inventory_state for _ in range(len(state_columns))]
        new_state_row = pd.DataFrame([inventory_state_list], columns=state_columns, index=["inventory_state"])
        new_state = state.append(new_state_row)
        return new_state
