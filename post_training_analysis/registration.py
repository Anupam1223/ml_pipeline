import importlib
import logging

ana_log = logging.getLogger("Analysis Registry")


def load(name):
    mod_name, attr_name = name.split(":")
    print(f'Attempting to load {mod_name} with {attr_name}')
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, attr_name)
    return fn


class AnaSpec(object):
    def __init__(self, id, entry_point=None, kwargs=None):
        self.id = id
        self.entry_point = entry_point
        self._kwargs = {} if kwargs is None else kwargs

    def make(self, **kwargs):
        """Instantiates an instance of data module with appropriate kwargs"""
        if self.entry_point is None:
            raise ana_log.error('Attempting to make deprecated analysis module {}. \
                               (HINT: is there a newer registered version \
                               of this analysis module?)'.format(self.id))
        _kwargs = self._kwargs.copy()
        _kwargs.update(kwargs)
        if callable(self.entry_point):
            gen = self.entry_point(**_kwargs)
        else:
            cls = load(self.entry_point)
            gen = cls(**_kwargs)

        return gen


class AnaRegistry(object):
    def __init__(self):
        self.ana_specs = {}

    def make(self, path, **kwargs):
        if len(kwargs) > 0:
            ana_log.info('Making new analysis module: %s (%s)', path, kwargs)
        else:
            ana_log.info('Making new analysis module: %s', path)
        ana_spec = self.spec(path)
        ana = ana_spec.make(**kwargs)

        return ana

    def all(self):
        return self.ana_specs.values()

    def spec(self, path):
        if ':' in path:
            mod_name, _sep, id = path.partition(':')
            try:
                importlib.import_module(mod_name)
            except ImportError:
                raise ana_log.error('A module ({}) was specified for the analysis module but was not found, \
                                   make sure the package is installed with `pip install` before \
                                   calling `analysis.make()`'.format(mod_name))

        else:
            id = path

        try:
            return self.ana_specs[id]
        except KeyError:
            raise ana_log.error('No registered analysis module with id: {}'.format(id))

    def register(self, id, **kwargs):
        if id in self.ana_specs:
            raise ana_log.error('Cannot re-register id: {}'.format(id))
        self.ana_specs[id] = AnaSpec(id, **kwargs)


# Global data registry
ana_registry = AnaRegistry()


def register(id, **kwargs):
    return ana_registry.register(id, **kwargs)


def make(id, **kwargs):
    return ana_registry.make(id, **kwargs)


def spec(id):
    return ana_registry.spec(id)

def list_registered_modules():
    return list(ana_registry.ana_specs.keys())
