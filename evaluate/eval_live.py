from evaluate.eval_portfolio import EvaluatePortfolio
import pickle as pkl
import paths


class EvalLive(EvaluatePortfolio):

    def act(self):

        potential_buys = []
        sells = []

        for grp in self.data:
            trade = grp["data"].to_dict("records")
            assert len(trade) == 1
            trade = trade[0]
            trade["ticker"] = grp["ticker"]
            if trade["actions"] == "hold":
                continue
            elif trade["actions"] == "buy" and trade["tradeable"]:
                potential_buys.append(trade)
            elif trade["actions"] == "sell" and trade["tradeable"]:
                sells.append(trade)

        self._handle_buys(potential_buys)
        self._handle_sells(sells)


load = False

if load:
    with open("state.pkl", "wb") as f:
        el = pkl.load(f)
else:
    model = "4-15_02---16_02-21"
    el = EvalLive(paths.models_path / model,
                  max_buy_output=0.859695881605148)
el.act()
print()

with open("state.pkl", "wb") as f:
    pkl.dump(el, f)
