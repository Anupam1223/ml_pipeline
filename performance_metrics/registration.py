import importlib
import logging

metrics_log = logging.getLogger("Performance Metrics Registry")


def load(name):
    mod_name, attr_name = name.split(":")
    print(f'Attempting to load {mod_name} with {attr_name}')
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, attr_name)
    return fn


class MetricsSpec(object):
    def __init__(self, id, entry_point=None, kwargs=None):
        self.id = id
        self.entry_point = entry_point
        self._kwargs = {} if kwargs is None else kwargs

    def make(self, **kwargs):
        """Instantiates an instance of the performance metrics with appropriate kwargs"""
        if self.entry_point is None:
            raise metrics_log.error('Attempting to make deprecated metrics {}. \
                               (HINT: is there a newer registered version \
                               of this metrics?)'.format(self.id))
        _kwargs = self._kwargs.copy()
        _kwargs.update(kwargs)
        if callable(self.entry_point):
            gen = self.entry_point(**_kwargs)
        else:
            cls = load(self.entry_point)
            gen = cls(**_kwargs)

        return gen


class MetricsRegistry(object):
    def __init__(self):
        self.metrics_specs = {}

    def make(self, path, **kwargs):
        if len(kwargs) > 0:
            metrics_log.info('Making new performance metrics: %s (%s)', path, kwargs)
        else:
            metrics_log.info('Making new performance metrics: %s', path)
        metrics_spec = self.spec(path)
        metrics = metrics_spec.make(**kwargs)

        return metrics

    def all(self):
        return self.metrics_specs.values()

    def spec(self, path):
        if ':' in path:
            mod_name, _sep, id = path.partition(':')
            try:
                importlib.import_module(mod_name)
            except ImportError:
                raise metrics_log.error('A module ({}) was specified for the metrics but was not found, \
                                   make sure the package is installed with `pip install` before \
                                   calling `metrics.make()`'.format(mod_name))

        else:
            id = path

        try:
            return self.metrics_specs[id]
        except KeyError:
            raise metrics_log.error('No registered metrics with id: {}'.format(id))

    def register(self, id, **kwargs):
        if id in self.metrics_specs:
            raise metrics_log.error('Cannot re-register id: {}'.format(id))
        self.metrics_specs[id] = MetricsSpec(id, **kwargs)


# Global metrics registry
metrics_registry = MetricsRegistry()


def register(id, **kwargs):
    return metrics_registry.register(id, **kwargs)


def make(id, **kwargs):
    return metrics_registry.make(id, **kwargs)


def spec(id):
    return metrics_registry.spec(id)

def list_registered_modules():
    return list(metrics_registry.metrics_specs.keys())
