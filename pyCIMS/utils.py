import re


def range_available(g, node, tech, add_upper=True):
    """
    NOTE: year not mentioned atm because dataset specifies that it remains constant

    Returns +1 at upper boundary if using range() function
    """
    avail = g.nodes[node]['2000']['technologies'][tech]['Available']['year_value']
    unavail = g.nodes[node]['2000']['technologies'][tech]['Unavailable']['year_value']
    if add_upper:
        return int(avail), int(unavail) + 1
    else:
        return int(avail), int(unavail)


def is_year(cn: str or int) -> bool:
    """ Determines whether `cn` is a year

    Parameters
    ----------
    cn : int or str
        The value to check to determine if it is a year.

    Returns
    -------
    bool
        True if `cn` is made entirely of digits [0-9] and is 4 characters in length. False otherwise.

    Examples
    --------
    >>> is_year(1900)
    True

    >>> is_year('2010')
    True
    """
    re_year = re.compile(r'^[0-9]{4}$')
    return bool(re_year.match(str(cn)))


def search_nodes(search_term, g):
    """
    Search `graph` to find the nodes which contain `search_term` in the node name's final component. Not case
    sensitive.

    Parameters
    ----------
    search_term : str
        The search term.

    Returns
    -------
    list [str]
        A list of node names (branch format) whose last component contains `search_term`.
    """
    def search(name):
        components = name.split('.')
        last_comp = components[-1]
        return search_term.lower() in last_comp.lower()

    return [n for n in g.nodes if search(n)]


def create_value_dict(year_val, source=None, branch=None, unit=None):
    value_dictionary = {'source': source,
                        'branch': branch,
                        'unit': unit,
                        'year_value': year_val
                        }

    return value_dictionary

