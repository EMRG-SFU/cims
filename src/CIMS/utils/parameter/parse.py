import re


def is_year(val: str | int) -> bool:
    """ Determines whether `cn` is a year

    Parameters
    ----------
    val : int or str
        The value to check to determine if it is a year.

    Returns
    -------
    bool
        True if `cn` is made entirely of digits [0-9] and is 4 characters in length. False
        otherwise.

    Examples
    --------
    >>> is_year(1900)
    True

    >>> is_year('2010')
    True
    """
    re_year = re.compile(r'^\d{4}$')

    return bool(re_year.match(str(val)))


def infer_type(d):
    """ 

    `d` is a value assumed to be a string.  
    We first try to see if represents a boolean value, and if it does
    we return the equivalent python boolean type. We then see if it contains a
    percent sign, and if it does we try convert it to the fractional representation
    of the percentage as a float (but simply return the string if that fails). We
    then see if it contains any commas, and if so we strip them out and again try
    parse as a float (assuming these are 'thousands' separators for legibility). If
    that fails we again just return the input string.  Finally, we try to parse it
    to a float that is returned, and if that fails, we just return the `d` input string
    unmolested.  

    """
    if (type(d) == str) and (d.lower() == "true"):
        return( True )

    elif (type(d) == str) and (d.lower() == "false"):
        return( False )

    elif (type(d) == str) and ('%' in d):
        try:
            return( float(d.replace("%","")) / 100.0)
        except:
            return( d )

    elif (type(d) == str) and (',' in d):
        try:
            return( float(d.replace(",","")) )
        except:
            return( d )

    else:
        try:
            return( float(d) )
        except:
            return( d )