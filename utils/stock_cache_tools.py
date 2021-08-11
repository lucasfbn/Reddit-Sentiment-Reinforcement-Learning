from preprocessing.price_data.cache import Cache
import paths

c = Cache(paths.price_data_cache)


def drop_tail(n):
    c.drop_tail(n)


drop_tail(7)
