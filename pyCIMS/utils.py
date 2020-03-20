import re
import logging

# Configure logging and set flag to raise exceptions
logging.raiseExceptions = True
logger = logging.getLogger(__name__)
logger.info('Start')


def get_name(branch_name):
    '''
    Fetch name of the object of interest from a branch name

    Parameters
    ----------
    :branch_name: str, name of branch

    Returns
    -------
    :name: str, name of last word to the right in branch name
    '''
    # name = branch_name.split('.')[-1:][0]
    # return name
    return branch_name


def split_unit(unit):
    '''
    split the name when unit is of form a/b
    '''
    units = unit.split('/')
    return units


def check_type(variable, datatype, node="unnamed", passing=False):
    try:
        if not isinstance(variable, datatype):
            raise TypeError
    except TypeError:
        logger.error(f"Error in node {get_name(node)}, \nData should be a {datatype} but is a {type(variable)}\n",
                      exc_info=False)
        if passing:
            pass
        else:
            raise


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


def get_nested_values(dict, key_list):
    """Retrieves value from the nested dictionary `dict` using the `key_list`.

    Parameters
    ----------
    dict : dict {str : dict or str}
        A nested dictionary, with at least `len(key_list)` levels of nesting.

    key_list : list [str]
        A list of keys used to access a value in dict. Where elements of `key_list` can be denoted as k0...kn,
        k0 must be a key in `dict`, k1 must be a key in `dict[k0]`, ... , kn must be a key in
        `dict[k0][k1]...[kn-1]`.

    Returns
    -------
    any
        Returns the value found by using the keys found in `key_list` to access a value within the nested
        dictionary `dict`. For example, if `key_list` = [x, y, z] then this function will retrieve the value
        found at `dict[x][y][z]`
    """
    value = dict
    for k in key_list:
        value = value[k]
    return value


def aggregate(sub_graph, agg_key, agg_func=sum):
    """
    Retrieves values to aggregate from each node in the `sub_graph` using `agg_key`. Then applies the `agg_func` to
    these values to get a final aggregation.
    Parameters
    ----------
    sub_graph : networkx.Graph
        The graph to be aggregated over.

    agg_key : list [str]
        The key list needed to access the values for aggregation.
        (ML! I need clarification here)

    agg_func : function (any) -> any
        The aggregation function to apply over the collection of values retreived using `agg_key` from all nodes in
        the `sub_graph`.

    Returns
    -------
    any
        The result from applying `agg_func` to the list of aggregated values retrieved from `sub_graph` using
        `agg_key`.
    """

    values_by_node = [get_nested_values(data, agg_key) for name, data in sub_graph.nodes(data=True)]

    all_values = [(k, v) for values in values_by_node for k, v in values.items()]

    # Make value lists, separated by key
    value_lists = {}
    for k, v in all_values:
        try:
            value_lists[k].append(v)
        except KeyError:
            value_lists[k] = [v]

    # Aggregate each list by applying agg_function
    aggregates = {k: agg_func(v) for k, v in value_lists.items()}

    return aggregates


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
