
class ModuleConfig:
    def __init__(self):
        self._data = {}

    def bind(self, **kwargs):
        self._data.update(kwargs)

    def get(self, key, default=None):
        return self._data.get(key, default)

    @property
    def all(self):
        return self._data.copy()

config = ModuleConfig()

