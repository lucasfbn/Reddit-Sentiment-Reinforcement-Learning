class Position:

    def __init__(self, ticker_name, price_raw, quantity):
        self.quantity = quantity
        self.price_raw = price_raw
        self.ticker_name = ticker_name


class Sim:
    BALANCE = 1000
    INVENTORY = []
    MAX_INVESTMENT_PER_TRADE = 50

    def buy(self, seq):
        quantity = self.MAX_INVESTMENT_PER_TRADE // seq.price_raw
        buy_price = quantity * seq.price_raw

        position = Position(ticker_name=seq.ticker_name,
                            price_raw=seq.price_raw,
                            quantity=quantity)

        self.INVENTORY.append(position)
        self.BALANCE -= buy_price

        reward = 0
        return reward

    def sell(self, seq):
        reward = 0
        new_inventory = []

        for pos in self.INVENTORY:
            if pos.ticker_name == seq.ticker_name:
                profit_loss = pos.quantity * seq.price_raw
                self.BALANCE += profit_loss
                reward += profit_loss
            else:
                new_inventory.append(pos)

        self.INVENTORY = new_inventory
        return reward

    def process_sequences(self, sequences):

        total_reward = 0

        for seq in sequences:
            if seq.action == 0:
                continue
            elif seq.action == 1:
                total_reward += self.buy(seq)
            elif seq.action == 2:
                total_reward += self.sell(seq)
            else:
                raise ValueError("Invalid action.")

        return total_reward


class MockObj(object):
    def __init__(self, **kwargs):  # constructor turns keyword args into attributes
        self.__dict__.update(kwargs)


day_0 = [
    MockObj(ticker_name="00", price_raw=10, action=1),
    MockObj(ticker_name="01", price_raw=10, action=1),
    MockObj(ticker_name="02", price_raw=10, action=1)
]

day_1 = [
    MockObj(ticker_name="00", price_raw=10, action=2),
    MockObj(ticker_name="01", price_raw=10, action=2),
    MockObj(ticker_name="02", price_raw=10, action=2)
]

sim = Sim()
print(sim.process_sequences(day_0))
print(sim.process_sequences(day_1))
