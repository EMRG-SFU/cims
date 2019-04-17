
from os import path
import uuid
from IPython.display import display_javascript, display_html, display
import json


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

    
