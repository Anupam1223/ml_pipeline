from mlflow_jlab.post_training_analysis.registration import register, make, list_registered_modules

register(
    id="punzi_analysis_v0",
    entry_point="mlflow_jlab.post_training_analysis.punzi_analysis:PunziAnalysis"
)

from mlflow_jlab.post_training_analysis.punzi_analysis import PunziAnalysis

register(
    id="primex_analysis_v0",
    entry_point="mlflow_jlab.post_training_analysis.primex_analysis:PrimexAnalysis"
)

from mlflow_jlab.post_training_analysis.primex_analysis import PrimexAnalysis

register(
    id="ensemble_analysis_v0",
    entry_point="mlflow_jlab.post_training_analysis.ensemble_analysis:EnsembleAnalysis"
)

from mlflow_jlab.post_training_analysis.ensemble_analysis import EnsembleAnalysis

register(
    id="jef_analysis_v0",
    entry_point="mlflow_jlab.post_training_analysis.jef_analysis:JefAnalysis"
)

from mlflow_jlab.post_training_analysis.jef_analysis import JefAnalysis

register(
    id="anupam_analysis_v0",
    entry_point="mlflow_jlab.post_training_analysis.anupam_analysis:AnupamAnalysis"
)

from mlflow_jlab.post_training_analysis.anupam_analysis import AnupamAnalysis

register(
    id="anupam_analysis_rf_v0",
    entry_point="mlflow_jlab.post_training_analysis.anupam_rf_analysis:RandomForestAnalysis"
)

from mlflow_jlab.post_training_analysis.anupam_rf_analysis import RandomForestAnalysis

register(
    id="anupam_analysis_bdt_v0",
    entry_point="mlflow_jlab.post_training_analysis.anupam_bdt_analysis:BoostedDecisionTreeAnalysis"
)

from mlflow_jlab.post_training_analysis.anupam_bdt_analysis import BoostedDecisionTreeAnalysis