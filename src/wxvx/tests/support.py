from copy import deepcopy
from typing import Any


def with_del(d: dict, *args: Any) -> dict:
    """
    Delete a value at a given chain of keys in a dict.

    :param d: The dict to update.
    :param args: One or more keys navigating to the value to delete.
    """
    new = deepcopy(d)
    p = new
    for key in args[:-1]:
        p = p[key]
    del p[args[-1]]
    return new


def with_set(d: dict, val: Any, *args: Any) -> dict:
    """
    Set a value at a given chain of keys in a dict.

    :param d: The dict to update.
    :param val: The value to set.
    :param args: One or more keys navigating to the value to set.
    """
    new = deepcopy(d)
    p = new
    if args:
        for key in args[:-1]:
            p = p[key]
        p[args[-1]] = val
    return new
