import importlib
import logging

wrapper_log = logging.getLogger("Model Wrapper Registry")


def load(name):
    mod_name, attr_name = name.split(":")
    print(f'Attempting to load {mod_name} with {attr_name}')
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, attr_name)
    return fn


class WrapperSpec(object):
    def __init__(self, id, entry_point=None, kwargs=None):
        self.id = id
        self.entry_point = entry_point
        self._kwargs = {} if kwargs is None else kwargs

    def make(self, **kwargs):
        """Instantiates an instance of the model wrapper with appropriate kwargs"""
        if self.entry_point is None:
            raise wrapper_log.error('Attempting to make deprecated model wrapper {}. \
                               (HINT: is there a newer registered version \
                               of this model wrapper?)'.format(self.id))
        _kwargs = self._kwargs.copy()
        _kwargs.update(kwargs)
        if callable(self.entry_point):
            gen = self.entry_point(**_kwargs)
        else:
            cls = load(self.entry_point)
            gen = cls(**_kwargs)

        return gen


class WrapperRegistry(object):
    def __init__(self):
        self.wrapper_specs = {}

    def make(self, path, **kwargs):
        if len(kwargs) > 0:
            wrapper_log.info('Making new model: %s (%s)', path, kwargs)
        else:
            wrapper_log.info('Making new model: %s', path)
        wrapper_spec = self.spec(path)
        wrapper = wrapper_spec.make(**kwargs)

        return wrapper

    def all(self):
        return self.wrapper_specs.values()

    def spec(self, path):
        if ':' in path:
            mod_name, _sep, id = path.partition(':')
            try:
                importlib.import_module(mod_name)
            except ImportError:
                raise wrapper_log.error('A module ({}) was specified for the model wrapper but was not found, \
                                   make sure the package is installed with `pip install` before \
                                   calling `model_wrapper.make()`'.format(mod_name))

        else:
            id = path

        try:
            return self.wrapper_specs[id]
        except KeyError:
            raise wrapper_log.error('No registered model with id: {}'.format(id))

    def register(self, id, **kwargs):
        if id in self.wrapper_specs:
            raise wrapper_log.error('Cannot re-register id: {}'.format(id))
        self.wrapper_specs[id] = WrapperSpec(id, **kwargs)


# Global wrapper registry
wrapper_registry = WrapperRegistry()


def register(id, **kwargs):
    return wrapper_registry.register(id, **kwargs)


def make(id, **kwargs):
    return wrapper_registry.make(id, **kwargs)


def spec(id):
    return wrapper_registry.spec(id)

def list_registered_modules():
    return list(wrapper_registry.wrapper_specs.keys())
