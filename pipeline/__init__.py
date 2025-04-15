from pipeline.registration import register, make, list_registered_modules


register(
    id="pipeline_v0",
    entry_point="pipeline.ml_model_pipeline:AnupamPipeline"
)

from pipeline.ml_model_pipeline import AnupamPipeline




