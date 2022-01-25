import orjson
import pandas as pd


class Saver:

    def __init__(self, fp):
        self.fp = fp

    def _process_single(self, obj):
        return dict(
            name=obj.name,
            exclude=obj.exclude,
            evl=obj.evl.to_dict(),
            sequences=[{"index": seq.index,
                        "metadata": seq.metadata.to_dict(),
                        "evl": seq.evl.to_dict()}
                       for seq in obj.sequences]
        )

    def dump(self, data):
        lst = []

        for obj in data:
            store = self._process_single(obj)
            lst.append(store)

        def c(o):
            if isinstance(o, pd.Period):
                return str(o)

        with open(self.fp, "wb") as f:
            f.write(orjson.dumps(lst, default=c))
