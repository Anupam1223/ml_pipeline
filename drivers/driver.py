import sys
import os

# Add the project root (one level up from drivers/) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pipeline as pipelines
from cfg.register_model import pipeline_config, model_config, data_config, metric_config, component_names, visualization_config, analysis_config
import torch

dev = "cpu"
if torch.cuda.is_available():
   dev = "cuda"

# Load the pipeline
pipeline = pipelines.make("pipeline_v0",
   pipeline_config = pipeline_config,
   data_config = data_config,
   model_config = model_config,
   metrics_config = metric_config,
   visualization_config = visualization_config,
   analysis_config = analysis_config,
   component_names = component_names,
   device = dev)

# And then run it:
pipeline.run()
