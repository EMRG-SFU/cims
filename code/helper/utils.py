
from datetime import datetime
from os import path
from pathlib import Path
import errno
import logging
import uuid
from IPython.display import display_javascript, display_html, display
import json

'''
Helper functions for saving info about model
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
