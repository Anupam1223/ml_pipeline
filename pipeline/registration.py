import importlib
import logging

pipe_log = logging.getLogger("Pipeline Registry")


def load(name):
    mod_name, attr_name = name.split(":")
    print(f'Attempting to load {mod_name} with {attr_name}')
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, attr_name)
    return fn


class PipeSpec(object):
    def __init__(self, id, entry_point=None, kwargs=None):
        self.id = id
        self.entry_point = entry_point
        self._kwargs = {} if kwargs is None else kwargs

    def make(self, **kwargs):
        """Instantiates an instance of the pipeline with appropriate kwargs"""
        if self.entry_point is None:
            raise pipe_log.error('Attempting to make deprecated pipeline {}. \
                               (HINT: is there a newer registered version \
                               of this pipeline?)'.format(self.id))
        _kwargs = self._kwargs.copy()
        _kwargs.update(kwargs)
        if callable(self.entry_point):
            gen = self.entry_point(**_kwargs)
        else:
            cls = load(self.entry_point)
            gen = cls(**_kwargs)

        return gen


class PipeRegistry(object):
    def __init__(self):
        self.pipe_specs = {}

    def make(self, path, **kwargs):
        if len(kwargs) > 0:
            pipe_log.info('Making new pipeline: %s (%s)', path, kwargs)
        else:
            pipe_log.info('Making new pipeline: %s', path)
        pipe_spec = self.spec(path)
        pipe = pipe_spec.make(**kwargs)

        return pipe

    def all(self):
        return self.pipe_specs.values()

    def spec(self, path):
        if ':' in path:
            mod_name, _sep, id = path.partition(':')
            try:
                importlib.import_module(mod_name)
            except ImportError:
                raise pipe_log.error('A module ({}) was specified for the pipwlinw but was not found, \
                                   make sure the package is installed with `pip install` before \
                                   calling `pipeline.make()`'.format(mod_name))

        else:
            id = path

        try:
            return self.pipe_specs[id]
        except KeyError:
            raise pipe_log.error('No registered model with id: {}'.format(id))

    def register(self, id, **kwargs):
        if id in self.pipe_specs:
            raise pipe_log.error('Cannot re-register id: {}'.format(id))
        self.pipe_specs[id] = PipeSpec(id, **kwargs)


# Global pipeline registry
pipe_registry = PipeRegistry()


def register(id, **kwargs):
    return pipe_registry.register(id, **kwargs)


def make(id, **kwargs):
    return pipe_registry.make(id, **kwargs)


def spec(id):
    return pipe_registry.spec(id)

def list_registered_modules():
    return list(pipe_registry.pipe_specs.keys())
