from post_training_analysis.registration import register, make, list_registered_modules




register(
    id="anupam_analysis_v0",
    entry_point="post_training_analysis.analysis:AnupamAnalysis"
)

from post_training_analysis.analysis import AnupamAnalysis

