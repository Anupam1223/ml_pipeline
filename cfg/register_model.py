anupam_dir = r"C:\Users\anupa\Desktop\ML_SETUP"


input_dir = r"C:\Users\anupa\Desktop\ML_SETUP"
training_file = "data.csv"
output_dir = r"models/"

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

    'feature_names': [
        'trk_e_fom', 'trk_m_fom', 'trk_pi_fom', 'trk_k_fom', 'trk_p_fom', #5
        'trk_e_p', 'trk_e_px', 'trk_e_py', 'trk_e_pz', #4
        'trk_dedx_cdc','trk_dedx_fdc', #2
        'bcal_e_o_p', 'fcal_e_o_p', #2
        'trk_N_cell', 'trk_rmsTime', 'trk_sigLong', 'trk_sigTrans', 'trk_bcal_e_preshower', 
        'trk_bcal_e_l2', 'trk_bcal_e_l3', 'trk_bcal_e_l4', #8
        'trk_Nblk', 'trk_E1E9', 'trk_E9E25', 'trk_SumU', 'trk_SumV',#5
        'trk_dedx_tof', 
        'trk_dedx_sc'
    ], #5

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
