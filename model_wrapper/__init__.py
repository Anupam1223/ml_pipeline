from model_wrapper.registration import register, make, list_registered_modules


register(
    id="anupam_net_v0",
    entry_point="model_wrapper.neural_nets:AnupamNet"
)

from model_wrapper.neural_nets import AnupamNet

