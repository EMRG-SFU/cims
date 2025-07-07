def find_first(items, pred=bool, default=None):
    """
    Find first item for that pred is True
    """
    return next(filter(pred, items), default)


def find_first_index(items, pred=bool):
    """
    Find index of first item for that pred is True
    """
    return find_first(enumerate(items), lambda kcn: pred(kcn[1]))[0]