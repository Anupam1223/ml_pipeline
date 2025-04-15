from mlflow_jlab.model_wrapper.registration import register, make, list_registered_modules

register(
    id="scikit_model_v0",
    entry_point="mlflow_jlab.model_wrapper.scikit_model_wrapper:ScikitModelWrapper"
)

from mlflow_jlab.model_wrapper.scikit_model_wrapper import ScikitModelWrapper

register(
    id="punzi_net_v0",
    entry_point="mlflow_jlab.model_wrapper.punzi_net:PunziNet"
)

from mlflow_jlab.model_wrapper.punzi_net import PunziNet

register(
    id="primex_net_v0",
    entry_point="mlflow_jlab.model_wrapper.primex_net:PrimexNet"
)

from mlflow_jlab.model_wrapper.primex_net import PrimexNet

register(
    id="jef_net_v0",
    entry_point="mlflow_jlab.model_wrapper.jef_net:JefNet"
)

from mlflow_jlab.model_wrapper.jef_net import JefNet

register(
    id="anupam_net_v0",
    entry_point="mlflow_jlab.model_wrapper.anupam_net:AnupamNet"
)

from mlflow_jlab.model_wrapper.anupam_net import AnupamNet

register(
    id="anupam_rf_v0",
    entry_point="mlflow_jlab.model_wrapper.anupam_rf:RandomForestWrapper"
)

from mlflow_jlab.model_wrapper.anupam_rf import RandomForestWrapper

register(
    id="anupam_bdt_v0",
    entry_point="mlflow_jlab.model_wrapper.anupam_bdt:BoostedDecisionTreeWrapper"
)

from mlflow_jlab.model_wrapper.anupam_bdt import BoostedDecisionTreeWrapper