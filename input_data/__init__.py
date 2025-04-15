from input_data.registration import register, make, list_registered_modules


register(
    id="anupam_data_parser_v0",
    entry_point="input_data.data_parser:AnupamDataParser"
)

from input_data.data_parser import AnupamDataParser
