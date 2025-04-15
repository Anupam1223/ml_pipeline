from performance_metrics.registration import register, make, list_registered_modules


register(
    id="anupam_metrics_v0",
    entry_point="performance_metrics.metrics:AnupamMetrics"
)

from performance_metrics.metrics import AnupamMetrics


