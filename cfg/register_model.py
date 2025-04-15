anupam_dir = "/work/halld/home/anupam/pinn-halld"

# input_dir = "/work/halld/home/ijaegle/particle-gun-non-linear-FCAL-17052024/"
# training_file = "combined-all-pos-charged-particles-all-tkin-1degree-corr.root"
# output_dir = "gluex-1-f29b-nn/"

# input_dir = "/work/halld/home/ijaegle/particle-gun-non-linear-FCAL-17052024/"
# training_file = "combined-all-pos-charged-particles-all-tkin-fcal-corr.root"
# output_dir = "gluex-all-f29b-nn/"

# input_dir = "/work/halld/home/ijaegle/particle-gun-non-linear-FCAL-17052024/"
# training_file = "combined-all-pos-charged-particles-all-tkin-26degree-corr.root"
# output_dir = "gluex-26-f29b-nn/"

# Load the dataset from the ROOT file
# input_dir = "/work/halld/home/gxproj9/particle-gun-non-linear-FCAL-18122024/root-files"
# training_file = "particle-pos-all-tkin-theta-6.0-deg.root"
# output_dir = "gluex-6-f29b-nn/"

# input_dir = "/work/halld/home/gxproj9/particle-gun-non-linear-FCAL-18122024/root-files"
# training_file = "particle-pos-all-tkin-theta-36.0-deg.root"
# output_dir = "gluex-36-18122024-5nn-f29b-nn/"

# input_dir = "/work/halld/home/gxproj9/particle-gun-non-linear-FCAL-19122024/root-files"
# training_file = "particle-pos-all-tkin-theta-36.0-deg.root"
# output_dir = "gluex-36-19122024-f29b-nn/"

# input_dir = "/work/halld/home/gxproj9/particle-gun-non-linear-FCAL-18122024/root-files"
# training_file = "particle-pos-all-tkin-theta-36.0-deg.root"
# output_dir = "gluex-test-f29b-nn/"

# input_dir = "/work/halld/home/gxproj9/particle-gun-non-linear-FCAL-27122024/root-files"
# training_file = "particle-pos-all-tkin-theta-10.0-deg.root"
# output_dir = "gluex-10-27122024-5nn-f29b-nn/"

# input_dir = "/volatile/halld/home/gxproj9/particle-gun-non-linear-FCAL-27122024/root-files"
# training_file = "particle-neg-all-tkin-theta-26.0-deg.root "
# output_dir = "gluex-26-27122024-5nn-f29b-nn/"

# input_dir = "/volatile/halld/home/ijaegle/particle-gun-non-linear-FCAL-03092024/root-files/neg"
# training_file = "combined-all-neg-charged-particles-all-tkin-26.0degree-corr.root"
# output_dir = "gluex-26-27122024-4nn-f29b-nn-neg/new/"

# input_dir = "/volatile/halld/home/ijaegle/particle-gun-non-linear-FCAL-03092024/root-files/neg"
# training_file = "combined-all-neg-charged-particles-all-tkin-26.0degree-corr.root"
# output_dir = "gluex-26-27122024-4nn-f29b-nn-neg/new/"

input_dir = "/volatile/halld/home/gxproj9/particle-gun-non-linear-FCAL-27122024/root-files"
training_file = "particle-pos-all-tkin-theta-26.0-deg.root"
output_dir = "gluex-26-27122024-4nn-f29b-nn-pos/new/"

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

    #trying for new features after feature correlation test for FCAL-17052024/
    # 'feature_names': [
    #     'trk_e_fom', 'trk_m_fom', 'trk_pi_fom', 'trk_k_fom', 'trk_p_fom', #5
    #     'trk_e_p', 'trk_m_p', 'trk_pi_p', 'trk_k_p', 'trk_p_p',#5
    #     'trk_e_sign', 'trk_m_sign', 'trk_pi_sign', 'trk_k_sign', 'trk_p_sign',#5
    #     'trk_bcal_e', 'trk_bcal_t', #2
    #     'trk_dedx_cdc',#1
    #     'trk_N_cell', 'trk_rmsTime','trk_sigLong', 'trk_sigTrans','trk_bcal_e_preshower',#5
    #     'trk_bcal_e_l2','trk_bcal_e_l3','trk_bcal_e_l4','bcal_e_o_p',#4
    #     'trk_E1E9', 'trk_E9E25','trk_SumU','trk_SumV', 'fcal_e_o_p'], #5

    # 'feature_names': [
    #     'trk_e_fom', 'trk_m_fom', 'trk_pi_fom', 'trk_k_fom', 'trk_p_fom', #5
    #     'trk_e_p', 'trk_m_p', 'trk_pi_p', 'trk_k_p', 'trk_p_p',#5
    #     #'trk_e_sign', 'trk_m_sign', 'trk_pi_sign', 'trk_k_sign', 'trk_p_sign',#5
    #     'trk_fcal_dx_min', 'trk_fcal_dy_min', 'trk_fcal_e', 'trk_fcal_t', 'trk_bcal_e', 'trk_bcal_t', #6
    #     'trk_dedx_cdc',#1
    #     'trk_N_cell', 'trk_rmsTime','trk_sigLong', 'trk_sigTrans','trk_bcal_e_preshower',
    #     'trk_bcal_e_l2','trk_bcal_e_l3','trk_bcal_e_l4','bcal_e_o_p',#9
    #     'trk_Nblk','trk_E1E9', 'trk_E9E25','trk_SumU','trk_SumV', 'fcal_e_o_p'],

    'feature_names': [
        'trk_e_fom', 'trk_m_fom', 'trk_pi_fom', 'trk_k_fom', 'trk_p_fom', #5
        #'trk_e_chisq','trk_m_chisq', 'trk_pi_chisq','trk_k_chisq','trk_p_chisq',
        #'trk_e_prob', 'trk_m_prob', 'trk_pi_prob','trk_k_prob','trk_p_prob',
        #'trk_e_p', 'trk_m_p', 'trk_pi_p', 'trk_k_p', 'trk_p_p', #5
        #'trk_e_px', 'trk_m_px', 'trk_pi_px', 'trk_k_px', 'trk_p_px', #5
        #'trk_e_py', 'trk_m_py', 'trk_pi_py', 'trk_k_py', 'trk_p_py', #5
        #'trk_e_pz', 'trk_m_pz', 'trk_pi_pz', 'trk_k_pz', 'trk_p_pz', #5
        'trk_e_p', 'trk_e_px', 'trk_e_py', 'trk_e_pz', #4
        #'trk_k_fom', 'trk_p_fom',
        #'trk_k_chisq','trk_p_chisq',
        #'trk_k_prob', 'trk_p_prob',
        #'trk_k_p', 'trk_p_p',
        #'trk_k_px', 'trk_p_px',
        #'trk_k_py', 'trk_p_py',
        #'trk_k_pz', 'trk_p_pz',
        #'trk_k_fom', 'trk_p_fom', 'trk_pi_fom',
        #'trk_k_chisq','trk_p_chisq',
        #'trk_k_prob', 'trk_p_prob',
        #'trk_k_p', 'trk_p_p', 'trk_pi_p',
        #'trk_k_px', 'trk_p_px', 'trk_pi_px',
        #'trk_k_py', 'trk_p_py', 'trk_pi_py',
        #'trk_k_pz', 'trk_p_pz', 'trk_pi_pz',
        #'trk_lfdc_e','trk_lfdc_p','trk_lfdc_pi','trk_lfdc_k','trk_lfdc_m',
        #'trk_lcdc_e','trk_lcdc_p','trk_lcdc_pi','trk_lcdc_k','trk_lcdc_m',
        #'trk_bcal_e', 'trk_bcal_t', #2
        #'trk_fcal_e', 'trk_fcal_t', #2
        'trk_dedx_cdc','trk_dedx_fdc', #2
        'bcal_e_o_p', 'fcal_e_o_p', #2
        'trk_N_cell', 'trk_rmsTime', 'trk_sigLong', 'trk_sigTrans', 'trk_bcal_e_preshower', 'trk_bcal_e_l2', 'trk_bcal_e_l3', 'trk_bcal_e_l4', #8
        'trk_Nblk', 'trk_E1E9', 'trk_E9E25', 'trk_SumU', 'trk_SumV',#5
        'trk_dedx_tof', # 'trk_flightime_tof', 'trk_pathlength_tof', 'trk_deltaxtohit_tof', 'trk_deltaytohit_tof', #5
        'trk_dedx_sc'#, #'trk_flightime_sc', 'trk_pathlength_sc', #3
    ], #5

    'decay_branches': ['eta-to-ggpi0', 'eta-to-pi0pi0pi0', 'eta-to-gg', 'gp-to-pi0pi0p', 
    'gp-to-pi0etap', 'omega-to-gpi0-traj-1.8', 'omega-to-gpi0-traj-6.4'],
    'target_luminosity': 58,
    'luminosity_per_branch': [16650, 45.4954, 45.4954, 2.48809, 2.48809, 61.3558, 61.3558],
    'luminosity_factor': 1.0,
    
    'target_names': ['s_label'],
    # 'label_mapping' : 
            # {
            #     'k+': 0,
            #     'mu+': 1,
            #     'p+': 2,
            #     'pi+': 3,
            #     'e+': 4
            # }
    # 'label_mapping' : {
    #             'k+': 0,
    #             'p+': 1,
    #             'pi+': 2
    #         }
    # 'label_mapping' : {
    #             'k-': 0,
    #             'mu-': 1,
    #             'p-': 2,
    #             'e-': 3
    #         }
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
    'n_epochs_anupam': 1000,
    'mon_epoch_anupam': 30,
    'read_epoch_anupam': 5,
    'anupam_learning_rate': 1E-3,
    'batch_size_anupam': 4096,
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
    # 'particle_classes':
        # {
        #     0:'k+',
        #     1:'mu+',
        #     2:'p+',
        #     3:'pi+',
        #     4:'e+'
        # },
    # 'particle_classes':{
    #         0:'k+',
    #         1:'p+',
    #         2:'pi+'
    #     },
    'particle_classes':{
            0:'k+',
            1:'mu+',
            2:'p+',
            3:'e+'
        },
    # 'prediction_labels' :['k_pred', 'm_pred', 'p_pred', 'pi_pred','e_pred']
    # 'prediction_labels' :['k_pred', 'p_pred', 'pi_pred']
    'prediction_labels' :['k_pred', 'm_pred', 'p_pred','e_pred']
}
