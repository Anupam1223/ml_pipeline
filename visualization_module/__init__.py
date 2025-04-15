from mlflow_jlab.visualization_module.registration import register, make, list_registered_modules

register(
    id="regression_visualizer_v0",
    entry_point="mlflow_jlab.visualization_module.regression_visualizer:Visualizer"
)

from mlflow_jlab.visualization_module.regression_visualizer import Visualizer

register(
    id="punzi_visualizer_v0",
    entry_point="mlflow_jlab.visualization_module.punzi_visualizer:PunziVisualizer"
)

from mlflow_jlab.visualization_module.punzi_visualizer import PunziVisualizer

register(
    id="primex_visualizer_v0",
    entry_point="mlflow_jlab.visualization_module.primex_visualizer:PrimexVisualizer"
)

from mlflow_jlab.visualization_module.primex_visualizer import PrimexVisualizer

register(
    id="jef_visualizer_v0",
    entry_point="mlflow_jlab.visualization_module.jef_visualizer:JefVisualizer"
)

from mlflow_jlab.visualization_module.jef_visualizer import JefVisualizer

register(
    id="anupam_visualizer_v0",
    entry_point="mlflow_jlab.visualization_module.anupam_visualizer:AnupamVisualizer"
)

from mlflow_jlab.visualization_module.anupam_visualizer import AnupamVisualizer

register(
    id="anupam_visualizer_rf_v0",
    entry_point="mlflow_jlab.visualization_module.anupam_visualizer_rf:RandomForestVisualizer"
)

from mlflow_jlab.visualization_module.anupam_visualizer_rf import RandomForestVisualizer

register(
    id="anupam_visualizer_bdt_v0",
    entry_point="mlflow_jlab.visualization_module.anupam_visualizer_bdt:BoostedDecisionTreeVisualizer"
)

from mlflow_jlab.visualization_module.anupam_visualizer_bdt import BoostedDecisionTreeVisualizer