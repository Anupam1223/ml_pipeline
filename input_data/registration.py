import importlib
import logging

data_log = logging.getLogger("Data Registry")


def load(name):
    mod_name, attr_name = name.split(":")
    print(f'Attempting to load {mod_name} with {attr_name}')
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, attr_name)
    return fn


class DataSpec(object):
    def __init__(self, id, entry_point=None, kwargs=None):
        self.id = id
        self.entry_point = entry_point
        self._kwargs = {} if kwargs is None else kwargs

    def make(self, **kwargs):
        """Instantiates an instance of data module with appropriate kwargs"""
        if self.entry_point is None:
            raise data_log.error('Attempting to make deprecated data module {}. \
                               (HINT: is there a newer registered version \
                               of this data module?)'.format(self.id))
        _kwargs = self._kwargs.copy()
        _kwargs.update(kwargs)
        if callable(self.entry_point):
            gen = self.entry_point(**_kwargs)
        else:
            cls = load(self.entry_point)
            gen = cls(**_kwargs)

        return gen


class DataRegistry(object):
    def __init__(self):
        self.data_specs = {}

    def make(self, path, **kwargs):
        if len(kwargs) > 0:
            data_log.info('Making new data module: %s (%s)', path, kwargs)
        else:
            data_log.info('Making new data module: %s', path)
        data_spec = self.spec(path)
        data = data_spec.make(**kwargs)

        return data

    def all(self):
        return self.data_specs.values()

    def spec(self, path):
        if ':' in path:
            mod_name, _sep, id = path.partition(':')
            try:
                importlib.import_module(mod_name)
            except ImportError:
                raise data_log.error('A module ({}) was specified for the data module but was not found, \
                                   make sure the package is installed with `pip install` before \
                                   calling `data.make()`'.format(mod_name))

        else:
            id = path

        try:
            return self.data_specs[id]
        except KeyError:
            raise data_log.error('No registered data module with id: {}'.format(id))

    def register(self, id, **kwargs):
        if id in self.data_specs:
            raise data_log.error('Cannot re-register id: {}'.format(id))
        self.data_specs[id] = DataSpec(id, **kwargs)


# Global data registry
data_registry = DataRegistry()


def register(id, **kwargs):
    return data_registry.register(id, **kwargs)


def make(id, **kwargs):
    return data_registry.make(id, **kwargs)


def spec(id):
    return data_registry.spec(id)

def list_registered_modules():
    return list(data_registry.data_specs.keys())
