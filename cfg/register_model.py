from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

anupam_dir = os.getenv("ANUPAM_DIR")
input_dir = os.getenv("INPUT_DIR")
training_file = os.getenv("TRAINING_FILE")
output_dir = os.getenv("OUTPUT_DIR")
# Parse feature list
feature_names = os.getenv("FEATURES").split(",")

# input_dir = "/volatile/halld/home/gxproj9/particle-gun-non-linear-FCAL-10122024/root-files"
# training_file = "particle-neg-all-tkin-theta-26.0-deg.root"
# output_dir = "gluex-26-27122024-4nn-f29b-nn-neg/new/"

pipeline_config = {
    'output_loc': anupam_dir + "/" + output_dir,
    'write_model_to_file': True,
    'store_dataframes': True
}

component_names = {
    'data_parser': 'anupam_data_parser_v0',
    'model': 'anupam_net_v0',
    'performance_metric': 'anupam_metrics_v0',
    'visualizer': 'anupam_visualizer_v0',
    'analysis': 'anupam_analysis_v0'
}

data_config = {
    'seed': 123,
    'data_path': input_dir + "/" + training_file,
    'output_loc': anupam_dir + "/" + output_dir,

    'feature_names': feature_names, #5

    'decay_branches': ['eta-to-ggpi0', 'eta-to-pi0pi0pi0', 'eta-to-gg', 'gp-to-pi0pi0p', 
    'gp-to-pi0etap', 'omega-to-gpi0-traj-1.8', 'omega-to-gpi0-traj-6.4'],
    'target_luminosity': 58,
    'luminosity_per_branch': [16650, 45.4954, 45.4954, 2.48809, 2.48809, 61.3558, 61.3558],
    'luminosity_factor': 1.0,
    
    'target_names': ['s_label'],
    'label_mapping' : 
            {
                'k+': 0,
                'mu+': 1,
                'p+': 2,
                'e+': 3
            }
}

metric_config = {
    "n_mass_hypotheses": 200
}

model_config = {
    'seed': 42,
    'n_inputs': 28,
    'n_outputs': 4,
    'anupam_optimizer': 'adam',
    'n_epochs_anupam': 100,
    'mon_epoch_anupam': 30,
    'read_epoch_anupam': 5,
    'anupam_learning_rate': 1E-3,
    'batch_size_anupam': 128,
    'architecture': [140, 20, 200, 110, 150, 50],
    'activation': ["relu", "relu", "relu", "relu", "relu","relu"],
    'output_activation': 'softmax',
    'show_model_response': True,
    'watch_gradients':True,
    'dropouts': [0.15]*6
}

visualization_config = {}

analysis_config = {
    'output_loc': anupam_dir + '/' + output_dir + '/models/',
    'input_dir_data': '/work/halld/home/anupam/pinn-halld/testing/', 
    'data_f_file_name': 'anupam-testing-data.feather', 

    'particle_classes':{
            0:'k+',
            1:'mu+',
            2:'p+',
            3:'e+'
        },
    'prediction_labels' :['k_pred', 'm_pred', 'p_pred','e_pred']
}
