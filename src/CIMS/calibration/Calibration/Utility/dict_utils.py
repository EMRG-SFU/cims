import re
from collections.abc import Mapping, Sequence, Iterable

def collect_dict_keys(structure):
    """
    Recursively walk through a nested structure of dicts and lists (or other
    sequences) and return a flat list containing *all* dictionary keys found.

    Parameters
    ----------
    structure : Any
        The input data – typically a dict, list, tuple, or a combination thereof.

    Returns
    -------
    list
        A list of keys (as they appear in the original objects). Duplicates are
        preserved because the same key may occur in different branches.
    """
    keys = []

    # If we’re looking at a mapping (dict‑like), record its keys and dive into values
    if isinstance(structure, Mapping):
        for k, v in structure.items():
            keys.append(k)          # store the key itself
            keys.extend(collect_dict_keys(v))   # recurse into the value

    # If it’s a sequence (list, tuple, etc.) but *not* a string/bytes, iterate over items
    elif isinstance(structure, Sequence) and not isinstance(structure, (str, bytes, bytearray)):
        for item in structure:
            keys.extend(collect_dict_keys(item))

    # Anything else (int, float, None, custom objects…) is ignored – it can’t contain dict keys
    return keys



def collect_dict_keys_fullPath(structure, _prefix=None):
    """
    Recursively walk through a nested structure of dicts and sequences,
    returning a flat list of dictionary keys prefixed with the hierarchy
    that led to them.

    Parameters
    ----------
    structure : Any
        The input data – typically a dict, list, tuple, or a combination thereof.
    _prefix : tuple[str|int], optional
        Internal helper used during recursion to accumulate the path components.
        Users should not pass this argument.

    Returns
    -------
    list[str]
        A list where each entry is a hierarchical key such as
        ``key1__key2__blah`` or ``4__7__blah``. Duplicate entries are kept
        because the same key may appear in different branches.
    """
    # Normalise the prefix to a tuple for easy concatenation
    if _prefix is None:
        _prefix = ()

    collected = []

    # ------------------------------------------------------------------
    # Mapping (dict‑like) – prepend the current key to the path and dive
    # ------------------------------------------------------------------
    if isinstance(structure, Mapping):
        for k, v in structure.items():
            # Build the new path that includes this key
            new_prefix = _prefix + (k,)
            # Store the fully‑qualified key
            collected.append("__".join(str(p) for p in new_prefix))
            # Recurse into the value, passing along the updated prefix
            collected.extend(collect_dict_keys_fullPath(v, new_prefix))

    # ---------------------------------------------------------------
    # Sequence (list, tuple, …) – use the element index as the next part
    # ---------------------------------------------------------------
    elif (
        isinstance(structure, Sequence)
        and not isinstance(structure, (str, bytes, bytearray))
    ):
        for idx, item in enumerate(structure):
            # Update the prefix with the list index
            new_prefix = _prefix + (idx,)
            # Recurse into the element; note that we do NOT add anything
            # to `collected` here because indices alone aren’t dict keys.
            collected.extend(collect_dict_keys_fullPath(item, new_prefix))

    # ------------------------------------------------------------------
    # Anything else (int, float, None, custom objects…) – nothing to do
    # ------------------------------------------------------------------
    return collected


def collect_dict_keys_fullPath_stopTech(structure, _prefix=None, stopNow=False):
    """

    This version does not iterate into a node's technology dict, if it has one. It prints the keys IN the tech
    dict, but doesn't recursively call itself into those. This is just to see if this is a less overwhelming view into the
    CIMS network of services.
    
    Recursively walk through a nested structure of dicts and sequences,
    returning a flat list of dictionary keys prefixed with the hierarchy
    that led to them.

    Parameters
    ----------
    structure : Any
        The input data – typically a dict, list, tuple, or a combination thereof.
    _prefix : tuple[str|int], optional
        Internal helper used during recursion to accumulate the path components.
        Users should not pass this argument.

    Returns
    -------
    list[str]
        A list where each entry is a hierarchical key such as
        ``key1__key2__blah`` or ``4__7__blah``. Duplicate entries are kept
        because the same key may appear in different branches.
    """
    # Normalise the prefix to a tuple for easy concatenation
    if _prefix is None:
        _prefix = ()

    collected = []

    # ------------------------------------------------------------------
    # Mapping (dict‑like) – prepend the current key to the path and dive
    # ------------------------------------------------------------------
    if isinstance(structure, Mapping):
        for k, v in structure.items():
            # Build the new path that includes this key
            new_prefix = _prefix + (k,)
            # Store the fully‑qualified key
            collected.append("__".join(str(p) for p in new_prefix))
            # Recurse into the value, passing along the updated prefix
            
            #print(f"{k} and {type(v)}")
            
            if stopNow == True:
                print("We are in the stopNow thing")
                pass
            elif isinstance(v, Mapping) and ('year_value' in list(v)):
                print("Next level down has 'year_value'. Stopping.")
                pass
            elif k == 'technologies' or k == 'price multiplier':
                print("We are in technologies thing")
                collected.extend(collect_dict_keys_fullPath_stopTech(v, new_prefix, True))
            else:
                print("We are in default thing")
                collected.extend(collect_dict_keys_fullPath_stopTech(v, new_prefix, False))

    # ---------------------------------------------------------------
    # Sequence (list, tuple, …) – use the element index as the next part
    # ---------------------------------------------------------------
    elif (
        isinstance(structure, Sequence)
        and not isinstance(structure, (str, bytes, bytearray))
    ):
        for idx, item in enumerate(structure):
            # Update the prefix with the list index
            new_prefix = _prefix + (idx,)
            # Recurse into the element; note that we do NOT add anything
            # to `collected` here because indices alone aren’t dict keys.
            collected.extend(collect_dict_keys_fullPath_stopTech(item, new_prefix, False))

    # ------------------------------------------------------------------
    # Anything else (int, float, None, custom objects…) – nothing to do
    # ------------------------------------------------------------------
    return collected



def omit_keys(orig_dict: dict, keys_to_remove: list[str]) -> dict:
    """ Return dict with a subset of original keys.
    From Lumo.
    Return a new dictionary that contains everything from ``orig_dict``
    except the keys listed in ``keys_to_remove``.

    Parameters
    ----------
    orig_dict: dict
        The source dictionary.
    keys_to_remove: list[str]
        Keys that should be omitted if they exist in ``orig_dict``.

    Returns
    -------
    dict
        A shallow copy of ``orig_dict`` without the unwanted keys.
    """
    # Convert the list to a set for O(1) membership tests.
    remove_set = set(keys_to_remove)

    # Dictionary comprehension builds the result in a single pass.
    return {k: v for k, v in orig_dict.items() if k not in remove_set}

