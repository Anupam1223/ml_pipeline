from mlflow_jlab.input_data.registration import register, make, list_registered_modules

register(
    id="numpy_pandas_data_v0",
    entry_point="mlflow_jlab.input_data.numpy_pandas_data:NumpyPandasData"
)

from mlflow_jlab.input_data.numpy_pandas_data import NumpyPandasData

register(
    id="punzi_data_parser_v0",
    entry_point="mlflow_jlab.input_data.punzi_data_parser:PunziDataParser"
)

from mlflow_jlab.input_data.punzi_data_parser import PunziDataParser

register(
    id="primex_data_parser_v0",
    entry_point="mlflow_jlab.input_data.primex_data_parser:PrimexDataParser"
)

from mlflow_jlab.input_data.primex_data_parser import PrimexDataParser


register(
    id="jef_data_parser_v0",
    entry_point="mlflow_jlab.input_data.jef_data_parser:JefDataParser"
)

from mlflow_jlab.input_data.jef_data_parser import JefDataParser

register(
    id="anupam_data_parser_v0",
    entry_point="mlflow_jlab.input_data.anupam_data_parser:AnupamDataParser"
)

from mlflow_jlab.input_data.anupam_data_parser import AnupamDataParser

register(
    id="anupam_rf_data_parser_v0",
    entry_point="mlflow_jlab.input_data.anupam_rf_data_parser:RandomForestDataParser"
)

from mlflow_jlab.input_data.anupam_rf_data_parser import RandomForestDataParser

register(
    id="anupam_bdt_data_parser_v0",
    entry_point="mlflow_jlab.input_data.anupam_bdt_data_parser:BoostedDecisionTreeDataParser"
)

from mlflow_jlab.input_data.anupam_bdt_data_parser import BoostedDecisionTreeDataParser