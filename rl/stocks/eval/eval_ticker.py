from tqdm import tqdm

from rl.utils.predict_proba import predict_proba


class Eval:

    def __init__(self, ticker, model, training_env_cls, trading_env_cls):
        self.trading_env_cls = trading_env_cls
        self.training_env_cls = training_env_cls
        self.model = model
        self.ticker = ticker

    def _eval_sequences(self, sequences):

        trading_env = self.trading_env_cls()

        for seq in sequences:
            state = self.training_env_cls.state_handler.forward(seq,
                                                                trading_env.inventory_state())
            action, proba = predict_proba(self.model, state)

            if action == 0:
                reward = trading_env.hold(seq.metadata.price)
            elif action == 1:
                reward = trading_env.buy(seq.metadata.price)
            elif action == 2:
                reward = trading_env.sell(seq.metadata.price)
            else:
                raise ValueError("Invalid action.")

            seq.evl.action = action
            seq.evl.reward = reward
            seq.evl.probas = proba
            seq.evl.open_positions = len(trading_env.inventory)
            seq.evl.split_probas()

        return len(trading_env.inventory)

    def eval_ticker(self):
        for ticker in tqdm(self.ticker):
            ticker.evl.open_positions = self._eval_sequences(ticker.sequences)

        _ = [(ticker.drop_sequence_data(), ticker.aggregate_rewards()) for ticker in self.ticker]
        return self.ticker
