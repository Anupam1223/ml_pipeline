import importlib
import logging

viz_log = logging.getLogger("Visualization Registry")


def load(name):
    mod_name, attr_name = name.split(":")
    print(f'Attempting to load {mod_name} with {attr_name}')
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, attr_name)
    return fn


class VizSpec(object):
    def __init__(self, id, entry_point=None, kwargs=None):
        self.id = id
        self.entry_point = entry_point
        self._kwargs = {} if kwargs is None else kwargs

    def make(self, **kwargs):
        """Instantiates an instance of data module with appropriate kwargs"""
        if self.entry_point is None:
            raise viz_log.error('Attempting to make deprecated viz module {}. \
                               (HINT: is there a newer registered version \
                               of this viz module?)'.format(self.id))
        _kwargs = self._kwargs.copy()
        _kwargs.update(kwargs)
        if callable(self.entry_point):
            gen = self.entry_point(**_kwargs)
        else:
            cls = load(self.entry_point)
            gen = cls(**_kwargs)

        return gen


class VizRegistry(object):
    def __init__(self):
        self.viz_specs = {}

    def make(self, path, **kwargs):
        if len(kwargs) > 0:
            viz_log.info('Making new viz module: %s (%s)', path, kwargs)
        else:
            viz_log.info('Making new viz module: %s', path)
        viz_spec = self.spec(path)
        viz = viz_spec.make(**kwargs)

        return viz

    def all(self):
        return self.viz_specs.values()

    def spec(self, path):
        if ':' in path:
            mod_name, _sep, id = path.partition(':')
            try:
                importlib.import_module(mod_name)
            except ImportError:
                raise viz_log.error('A module ({}) was specified for the viz module but was not found, \
                                   make sure the package is installed with `pip install` before \
                                   calling `viz.make()`'.format(mod_name))

        else:
            id = path

        try:
            return self.viz_specs[id]
        except KeyError:
            raise viz_log.error('No registered data module with id: {}'.format(id))

    def register(self, id, **kwargs):
        if id in self.viz_specs:
            raise viz_log.error('Cannot re-register id: {}'.format(id))
        self.viz_specs[id] = VizSpec(id, **kwargs)


# Global data registry
viz_registry = VizRegistry()


def register(id, **kwargs):
    return viz_registry.register(id, **kwargs)


def make(id, **kwargs):
    return viz_registry.make(id, **kwargs)


def spec(id):
    return viz_registry.spec(id)

def list_registered_modules():
    return list(viz_registry.viz_specs.keys())
