from rl.envs.simple_trading import SimpleTradingEnv


class SimpleTradingEnvEvaluation:

    def __init__(self, ticker):
        self.ticker_name = ticker.name
        self.sequences = ticker.sequences

        self._map_actions()

        self._run_env_result = None
        self._get_figure_points_result = None

    def _map_actions(self):
        map_ = {0: "hold", 1: "buy", 2: "sell"}
        for seq in self.sequences:
            seq.action = map_[seq.action]

    def run_env(self):
        ste = SimpleTradingEnv(self.ticker_name)

        # Realistic mode
        ste.ENABLE_TRANSACTION_COSTS = True
        ste.TRANSACTION_FEE_ASK = 0.05
        ste.TRANSACTION_FEE_BID = 0.01
        ste.ENABLE_NEG_BUY_REWARD = True
        ste.ENABLE_POS_SELL_REWARD = True

        # Tracker
        N_holds = 0
        N_buys = 0
        N_sells = 0
        N_positive_trades = 0
        N_negative_trades = 0
        profit = 0

        for seq in self.sequences:
            price = seq.price

            # Hold
            if seq.action == "hold":
                reward = ste.hold(price)
                N_holds += 1

            # Buy
            elif seq.action == "buy":
                reward = ste.buy(price)
                N_buys += 1

            # Sell
            elif seq.action == "sell":
                reward = ste.sell(price)
                N_sells += 1

                if reward - price > 0:
                    N_positive_trades += 1
                else:
                    N_negative_trades += 1

            else:
                raise ValueError("Invalid action")

            profit += reward

        N_open_positions = len(ste.inventory)
        B_open_positions = 1 if len(ste.inventory) > 0 else 0
        N_total_trades = N_holds + N_buys + N_sells

        self._run_env_result = {
            "N_holds": N_holds,
            "N_buys": N_buys,
            "N_sells": N_sells,
            "N_positive_trades": N_positive_trades,
            "N_negative_trades": N_negative_trades,
            "profit": profit,
            "N_open_positions": N_open_positions,
            "B_open_positions": B_open_positions,
            "N_total_trades": N_total_trades,
        }
        return self._run_env_result

    def get_pred_history(self):
        time, price, actions = [], [], []

        for i, seq in enumerate(self.sequences):
            time.append(i)
            price.append(seq.price)
            actions.append(seq.action)

        self._get_figure_points_result = {"time": time, "price": price, "actions": actions}
        return self._get_figure_points_result

    def to_dict(self):
        if self._run_env_result is None:
            self.run_env()
        if self._get_figure_points_result is None:
            self.get_pred_history()

        return {
            "ticker": self.ticker_name,
            "eval": self._run_env_result,
            "predicted": self._get_figure_points_result
        }


class AccumulatedSimpleTradingEnvEvaluation:

    def __init__(self, pred_ticker):
        self.pred_ticker = pred_ticker
        self.run_stats = []
        self.totals = {}

    def run_evaluation(self):
        for ticker in self.pred_ticker:
            stee = SimpleTradingEnvEvaluation(ticker)
            stee.run_env()
            stee.get_pred_history()
            self.run_stats.append(stee.to_dict())
        return self.run_stats

    def accumulate_evaluation(self):
        # Dict from keys
        totals = {key: 0 for key in self.run_stats[0]["eval"]}

        for run in self.run_stats:
            for key, value in run["eval"].items():
                totals[key] += value
        self.totals = totals
        return totals

    def to_dict(self):
        temp = {}
        temp["totals"] = self.totals.copy()
        temp["run_stats"] = self.run_stats
        return temp
