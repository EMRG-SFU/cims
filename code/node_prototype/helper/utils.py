
from datetime import datetime
from os import path
from pathlib import Path
import errno
import logging
import uuid
from IPython.display import display_javascript, display_html, display
import json

'''
Helper functions for pycims
'''

'''
HELPER FUNCTIONS TO NAVIGATE THROUGH THE NESTED LAYERS MORE EASILY
USAGE: FEED IN A FUNCTION TO FILTER OUT SOME KEY/VALUE PAIRS. THE FILTER FUNCTION TAKES A (ANONYMOUS/LAMBDA) FUNCTION
'''


def process_subtypes( data, func, callback = None, results = None, subcat = "subtypes" ):
    '''
    Function that navigates through the dictionary and takes a function to manipulate a subset of the data
    '''
    if results is None:
        results = {}
    if isinstance( data, list ) :
        for item in data:
            process_subtypes( data = item, func = func, subcat = subcat, callback = callback, results = results )
        return results
    
    elif isinstance( data, dict ):
        if "leaf" in data.keys() :
            if data["leaf"] :
                # add error checking for no "subcat" key 
                for key, value in data[subcat].items() :
                    if callback is not None:
                        func( key, value, results, callback )
                    else:
                        func(key, value, results)
                return results
        # if we got here, leaf is false, so go through the rest of the values
        for key, value in data.items():
            results = process_subtypes(data = value, func = func, subcat = subcat, callback = callback, results = results )
        return results
    else:
        # if we're asked to process a non-list/non-dict, then we don't have anything to do, so return 
        return results
    


def filter_func(key, value, results, condition_func):
    '''
    Args:
      * condition_func: lambda function that accepts a key/value pair and returns True or False
    returns a list of dicts with values that meet condition  
    '''
    # Iterate over all the items in dictionary
    if condition_func(value):        
        results[key] = value
    return results


def filter_obs(data, fuel_type = "electricity", val = "fuel", verbose=True):
    

    cond_func = lambda value: value["fuel"] == fuel_type
    filtered = process_subtypes( data["demand_step"], filter_func, cond_func )
    
    if verbose:
        print("Displaying all the techs that demands {}".format(fuel_type))
        for key, value in filtered.items():
            print("\n", key, value)
    
    return filtered

'''
# EXAMPLE USE FOR FUNCTIONS ABOVE

cond_func = lambda value: value["fuel"] == "electricity"
filtered = process_subtypes( techs, filter_func, cond_func )
'''

def select_subcategory( data, filter_func, subcat = "subtypes", results = None  ):
    '''
    Function to produce a *list* of key/values from iterating over the data and
    looking for a given subcategory (this assumes that the subcategory given is 
    a dictionary) (like with 'subtype' in the test data ) - 
    the resulting list is a series of 2-tuples with the key in the first member
    of the tuple and the value in the second
    (note: only looks in leaves for the subcategory)
    '''
    if results is None:
        results = []

    if isinstance( data, list ) :
        for item in data:
            select_subcategory( data = item, 
                                filter_func = filter_func, 
                                subcat = subcat, 
                                results = results )
        return results
    elif isinstance( data, dict ):
        # see if we're a leaf
        if "leaf" in data.keys() :
            if data["leaf"] :
                # see if we have the subcategory
                if subcat in data.keys():
                    for key, value in data[subcat].items() :
                        if filter_func( key, value ):
                            results.append( (key, value) )
                return results
        # if we got here, leaf is false, so go through the rest of the values
        for key, value in data.items():
            results = select_subcategory(data = value, 
                                         filter_func = filter_func, 
                                         subcat = subcat, 
                                         results = results )
        return results
    else:
        # if we're asked to process a non-list/non-dict, then we don't have anything to do, so return 
        return results

# time to get some closure
def value_includes( value_criteria ):
    # add error checking to make sure value_criteria is a dictionary
    def filter(test_key, test_value):
        # TODO: come up with better variable names
        for criteria_key, criteria_value in value_criteria.items():
            if criteria_key in test_value.keys():
                if test_value[criteria_key] != criteria_value:
                    return False
            else:
                return False
        # if we got here, every check was succesful
        # note: an empty value criteria will always be true
        return True

    # we now return a new function that we just made with the specific
    #  values filed in -- this is a closure or a more proper 'lambda'
    return filter

#  for the result of a select - return a list only of the 'values'
#  given that select returned both key and value
#  TODO: better name for this function
def reduce_to_values( select_results ):
    return [i[0] for i in select_results]

def gather_values( select_results, value_name ):
    # add error checking to make sure this is a list
    gathered_values = []
    for result in select_results:
        # see if this is a tuple
        if isinstance(result, tuple):
            # more error checking to make sure that result[1] is a dictionary
            if value_name in result[1]:
                gathered_values.append(result[1][value_name])
            else:
                # if the value is not there, we'll just stick a 'none' in
                # the list, probably not the behavior you want, but
                # it does a *something* now
                gathered_values.append( None )
        else:
            # we're going to assume it just a dictionary (say from reduce_to_values)
            # more error checking needed here
            if value_name in result:
                gathered_values.append(result[value_name])
            else:
                # as above
                gathered_values.append( None )
    return gathered_values


# the MOST IMPORTANT thing to note is that when do a select, we get back a list
#  and because of the way python works with *labels*... we get *labels* to 
#  the dictionaries in the original nested structure -- so when we change values
#  in the list, we change them in the original data structure
# Make sure you understand why this works!

def modify_values( select_results, value_key_name, new_value_key_value ):
    
    # NOTE: if you pass in a dictionary (or even list) as new_value_key_value it 
    #       probably won't do what you expect (labels again) -- you may want to 
    #       check for this and do a deep copy each time you insert it
    
    # for fun we'll return the number of values we've inserted, so you can 
    #  make sure that your operation has actually done something
    num_changes = 0
    
    # add error checking to make sure select_results is a list
    for result in select_results: 
        # see if this is a tuple
        if isinstance(result, tuple):
            # more error checking to make sure that result[1] is a dictionary
            if value_key_name in result[1]:
                result[1][value_key_name] = new_value_key_value
                num_changes += 1
            # if the value is not there, we don't do anything
        else:
            # we're going to assume it just a dictionary (say from reduce_to_values)
            # more error checking needed here
            if value_key_name in result:
                result[value_key_name] = new_value_key_value
                num_changes += 1
            # otherwise as above
    return num_changes



def find_price_for_fuel( fuel_type, data ):
    results = select_subcategory(data = data,
                                 filter_func = value_includes({"fuel": fuel_type}))
    result_values = gather_values( results, "price" )

    # now given that they're all the same, we can just look at the first one
    if len(result_values) == 0:
        found_price = None
    else:

        found_price = result_values[0]
    return found_price


def find_quantity_for_fuel( fuel_type, data ):
    results = select_subcategory(data = data,
                                 filter_func = value_includes({"fuel": fuel_type}))
    result_values = gather_values( results, "demand" )

    # now given that they're all the same, we can just look at the first one
    if len(result_values) == 0:
        found_quant = None
    else:

        found_quant = result_values
    return found_quant


def change_price_for_fuel( data, fuel_type, new_price ):
    results = select_subcategory(data = data,
                             filter_func = value_includes({"fuel": fuel_type}))

    num_changed = modify_values( results, "price", new_price)

#     if num_changed > 0:
#         print("Changed price of " + fuel_type + " to " + str(new_price))
#     else:
#         print("Unable to change price of " + fuel_type + ": no records")


def change_quantity_demanded( data, fuel_type, new_demand ):
    results = select_subcategory(data = data,
                             filter_func = value_includes({"fuel": fuel_type}))

    num_changed = modify_values( results, "demand", new_demand)



'''
SAVING FUNCTIONS
'''
def time_filename():
    '''
    Used to timestamp filenames

    Returns str of date and time
    '''
    time_file = datetime.now()
    time = '{}h{}m{}s'.format(time_file.hour,
                                         time_file.minute,
                                         time_file.second)
    date = '{}-{}-{}'.format(time_file.year,
                             time_file.month,
                             time_file.day)
    return time, date


def save_folder_file(save_dir, filename, ext='', optional_folder=''):
    '''
    Creates a string with a precise path to save files with time and dates.
    (using time_filename() fun above)
    Creates new directories if folders given do not exist

    Arguments:
    `save_dir`:        str, name of folder (child of current directory)
    `filename`:        str, a file name without extension. If empty str, creates a filename
    `ext`:             str, file extension (in .ext form)
    `optional_folder`: str, optional sub folder, child of save_dir

    Returns: str, full path
    '''
    time, date = time_filename()
    filename = filename + time + ext
    dirs = path.join(save_dir, optional_folder, date)
    full_path = path.join(dirs, filename)
    directory = Path(dirs)

    try:
        if not directory.exists():
            print('Directory does not exist, '\
                  'creating a new directory named /{}/...\n'.format(dirs))
            directory.mkdir(exist_ok=True, parents=True)

    except OSError as error:
        if error.errno != errno.EEXIST:
            raise

    return full_path

'''
LOGGING FUNCTIONS
'''
### DEBUGGING TOOLS

def init_logger(filename="supply_demand"):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    pathname = logged(filename=filename)
    file_handler = logging.FileHandler(pathname, mode="w")
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
#     return logger


def logged(save_dir='log', filename=''):
    '''
       Produces a log file. Creates a directory if save_dir does not exists
       * Required * for convergence plot (loglik vs iter) with LDA.

     Arguments:
      `save_dir`:        str, name of folder (child of current directory)
      `filename`:        str, a file name without extension. If empty str, creates a filename
     Returns: file path to use for plotting
    '''
    full_path_log = save_folder_file(save_dir=save_dir, filename=filename, ext='.log')

    logging.basicConfig(filename=full_path_log,
                        format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)

    return full_path_log



class RenderJSON:
    '''
    This is for a nice collapsable rendering of data for Jupyter notebook
    (see https://www.reddit.com/r/IPython/comments/34t4m7/lpt_print_json_in_collapsible_format_in_ipython/)
    Args:
     `json_data`: dict, the data from a json file (can be dict of dicts of dicts etc...)

    NOTE: For the moment, only works with internet connection
    '''
    def __init__(self, json_data):
        if isinstance(json_data, dict):
            self.json_str = json.dumps(json_data)
        else:
            self.json_str = json_data
        self.uuid = str(uuid.uuid4())

    def _ipython_display_(self):
        display_html('<div id="{}" style="height: 600px; width:100%;"></div>'.format(self.uuid), raw=True)
        display_javascript("""
        require(["https://rawgit.com/caldwell/renderjson/master/renderjson.js"], function() {
        document.getElementById('%s').appendChild(renderjson(%s))
        });
        """ % (self.uuid, self.json_str), raw=True)




class FindKey(dict):
    '''
    Example use:
        FindKey(d).get("region.Alberta.sector")
    will give the value associated with that path of keys in the dict

    '''
    def get(self, keypath, default=None):
        keys = keypath.split(".")
        val = None
        for key in keys:
            if val:
                if isinstance(val, list):
                    val = [v.get(key, default) if v else None for v in val]
                else:
                    val = val.get(key, default)
            else:
                val = dict.get(self, key, default)
            if not val:
                break
        return val
