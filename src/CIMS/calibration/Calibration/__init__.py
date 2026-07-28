from .config import config as _config

def bind_data(**kwargs):
    _config.bind(**kwargs)

def getData(key=None, default=None):
    return _config.get(key, default) if key else _config.all
