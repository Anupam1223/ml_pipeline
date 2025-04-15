from visualization_module.registration import register, make, list_registered_modules



register(
    id="anupam_visualizer_v0",
    entry_point="visualization_module.visualizer:AnupamVisualizer"
)

from visualization_module.visualizer import AnupamVisualizer