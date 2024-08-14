from collections.abc import Iterable

agg_float_func = min


def agg_float(data: Iterable[float], **kwargs) -> float:
    return agg_float_func((x for x in data), **kwargs)
