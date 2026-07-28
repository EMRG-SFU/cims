
def intersect_sublists(list_of_lists):
    """
    Return a list with the intersection of all strings found in the nested lists.

    Parameters
    ----------
    list_of_lists : list[list[str]]
        Example: [["apple", "banana", "cherry"],
                  ["banana", "cherry", "date"],
                  ["cherry", "banana"]]

    Returns
    -------
    list[str]
        Strings that are present in *all* sub‑lists.
        Order follows the first sub‑list (you can sort later if you prefer).
    """
    if not list_of_lists:                 # empty input → empty result
        return []

    # Start with the set of the first sub‑list
    common = set(list_of_lists[0])

    # Intersect with each subsequent sub‑list
    for sublist in list_of_lists[1:]:

        # Lumo came up with this one. The explanation says that for sets, & means intersection, and the compound
        # operator with = just updates the set in place.
        common &= set(sublist)            # same as common = common.intersection(set(sublist))

        # Early exit: if nothing is common any more we can stop
        if not common:
            return []

    # Preserve the order from the first sub‑list (optional)
    return [item for item in list_of_lists[0] if item in common]



def union_of_sublists(list_of_lists):
    """
    Return a list with the union of all strings found in the nested lists.

    Parameters
    ----------
    list_of_lists : list[list[str]]
        Example: [["apple", "banana"], ["banana", "cherry"], ["date"]]

    Returns
    -------
    list[str]
        A list of the distinct strings, order is preserved by first appearance.
    """
    seen = set()          # tracks strings we’ve already added
    result = []           # final union list

    for sublist in list_of_lists:
        for item in sublist:
            if item not in seen:
                seen.add(item)
                result.append(item)

    return result

