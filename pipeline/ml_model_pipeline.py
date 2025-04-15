from core.pipeline_core import Pipeline
import input_data as data_parsers
import model_wrapper as wrappers
import performance_metrics as metrics
import visualization_module as viz
import post_training_analysis as ana
from utils.config_reader import ConfigReader
import matplotlib.pyplot as plt
import os
import argparse
import torch

class AnupamPipeline(Pipeline):

    # Initialize:
    #**********************************
    def __init__(self,pipeline_config,data_config,model_config,metrics_config,visualization_config,analysis_config,component_names,device):
        self.pipeline_config = pipeline_config
        self.data_config = data_config
        self.model_config = model_config
        self.metrics_config = metrics_config
        self.visualization_config = visualization_config
        self.analysis_config = analysis_config
        self.component_names = component_names

        # Allow to overwrite certain config settings:
        parser = argparse.ArgumentParser(prog='ml_model_pipeline.py',description='Train and register Anupam Net')
        # General:
        parser.add_argument("--output_loc", help="Directory to save results", type=str, default=None)
        parser.add_argument('--write_model_to_file',action='store_true',default=None,help='Write model to file')
        parser.add_argument('--device',help="Torch Device (CPU , GPU)", type=str, default=None)
        parser.add_argument('--store_dataframes',help="Store dataframes",action='store_true', default=None)

        # Model:
        parser.add_argument("--n_epochs_anupam", help="Number of epochs for Anupam training", type=int, default=None)
        parser.add_argument("--mon_epoch_anupam", help="Report Anupam results every ith epoch", type=int, default=None)
        parser.add_argument("--read_epoch_anupam", help="Record Anupam results every ith epoch", type=int, default=None)
        parser.add_argument("--anupam_optimizer", help="Optimizer for Anupam training", type=str, default=None)
        parser.add_argument("--anupam_learning_rate", help="Learning for Anupam training", type=float, default=None)
        parser.add_argument("--watch_gradients",action='store_true',help="Monitor Gradients",default=None)

        # Monitoring
        parser.add_argument("--disable_roc_curve",action='store_true',help="Turn ROC curve off",default=None)
        parser.add_argument("--disable_threshold_scan",action='store_true',help="Turn threshold scan off",default=None)
        
        new_args = parser.parse_args()
        self.dev = device

        if new_args.device is not None:
            self.dev = new_args.device

        # Do the overwrite:
        self.overwrite_config(self.pipeline_config,'output_loc',new_args.output_loc)
        self.overwrite_config(self.pipeline_config,'write_model_to_file',new_args.write_model_to_file)
        self.overwrite_config(self.pipeline_config,'store_dataframes',new_args.store_dataframes)
        self.overwrite_config(self.model_config,'n_epochs_anupam',new_args.n_epochs_anupam)
        self.overwrite_config(self.model_config,'mon_epoch_anupam',new_args.mon_epoch_anupam)
        self.overwrite_config(self.model_config,'read_epoch_anupam',new_args.read_epoch_anupam)
        self.overwrite_config(self.model_config,'anupam_optimizer',new_args.anupam_optimizer)
        self.overwrite_config(self.model_config,'anupam_learning_rate',new_args.anupam_learning_rate)
        self.overwrite_config(self.model_config,'watch_gradients',new_args.watch_gradients)
        self.overwrite_config(self.visualization_config,'disable_roc_curve',new_args.disable_roc_curve)
        self.overwrite_config(self.visualization_config,'disable_threshold_scan',new_args.disable_threshold_scan)

        # Register settings for the pipeline:
        self.config_reader = ConfigReader(pipeline_config)
        self.output_loc = self.config_reader.load_setting("output_loc","training_results")
        self.model_store_path = self.output_loc+'/models'

        self.dataframes = None
        if self.pipeline_config['store_dataframes']:
          self.dataframes = self.output_loc+'/dataframes'

        self.write_model_to_file = self.config_reader.load_setting("write_model_to_file",False)
        if self.write_model_to_file:
            self.model_config['model_store_path'] = self.model_store_path

        # Get a nice intro:
        print("  ")
        print("**************************")
        print("*                        *")
        print("*   NEURAL-NET Registry  *")
        print("*                        *")
        print("**************************")
        print("  ")

    #**********************************

    # Overwrite config, if required:
    #**********************************
    def overwrite_config(self,config,setting,new_setting):
        if new_setting is not None:
            config[setting] = new_setting
    #**********************************

    # Load the individual components:
    #**********************************
    def load_components(self):
        # Data parser:
        self.data_parser = data_parsers.make(self.component_names['data_parser'],config=self.data_config,device=self.dev) 
        # Metrics:
        self.metric = metrics.make(self.component_names['performance_metric'],config=self.metrics_config,device=self.dev)
        metrics_dict = self.metric.return_metrics()
        # Register the anupam metrics for the anupam net:
        self.model_config['anupam_loss_fn'] = metrics_dict['anupam_loss']
        # Model:
        self.model = wrappers.make(self.component_names['model'],config=self.model_config,device=self.dev)
        # Visualization:
        self.visulaizer = viz.make(self.component_names['visualizer'],config=self.visualization_config)
        # Post training analysis:
        # We copy the feature names from the data parser, just to ensure that we are consistent with everything:
        self.analysis_config['feature_names'] = self.data_parser.feature_names
        self.analysis_config['anupam_softmax_fn'] = metrics_dict['softmax_loss']
        self.analysis = ana.make(self.component_names['analysis'],config=self.analysis_config,device=self.dev)
    #**********************************

    # Run everything together:
    #**********************************
    def run(self):
       # Load the components:
       print("Load pipeline components...")
    
       self.load_components()

       print("...done!")
       print("  ")

       # Create a result folder (if not already existing)
       if os.path.exists(self.output_loc) == False:
           print("Create folder to store training results and model...")

           os.mkdir(self.output_loc)

           # Check if a model path already exists:
           if os.path.exists(self.model_store_path) == False:
               os.mkdir(self.model_store_path)
           
           if self.dataframes is not None:
              # Check if dataframe path already exists:
              if os.path.exists(self.dataframes) == False:
                 os.mkdir(self.dataframes)

           print("...done!")
           print("  ")


       # Set up the jef model:
       print("Set up model...")

       self.model.build()

       print("...done!")
       print("  ")

       # Load the data:
       print("Load and prepare data...")

       data = self.data_parser.return_data()

       x_train = data['training_inputs']
       y_train = data['training_targets']
       w_train = data['weights_train']
    #    s_train = data['training_sigma_components']
       x_val = data['validation_inputs']
       y_val = data['validation_targets']
       w_val = data['weights_test']
    #    s_val = data['validation_sigma_components']

       print("...done!")
       print("  ")

       # Run the training:
       print("Train model...")

       loss_dict = self.model.fit(
           x=x_train,
           y=y_train,
           w=w_train,
           s=None,
           x_val=x_val,
           y_val=y_val,
           w_val=w_val,
           s_val=None)

       print("...done!")
       print("  ")
 
       # Diagnose the model and visualize it
       print("Run model diagnostics and visualize results...")

    #    x_label_bce = 'Epochs per ' + str(self.model_config['read_epoch_bce'])
       x_label_anupam = 'Epochs per ' + str(self.model_config['read_epoch_anupam'])

       data_labels = y_train
       if y_val is not None:
           data_labels = y_val
              
       plot_dict = self.visulaizer.visualize(loss_dict,x_label_anupam,data_labels)
              
       print("...done!")
       print("  ")     

       # Run the post training analysis:
       print("Perform post training analysis...")
       ana_plots = self.analysis.analyze(self.model,self.data_parser.df_dict,'anupam_prediction',self.dataframes)
       

       print("...done!")
       print("  ")     


       # Write everything to file:
       print("Write results and model to file...")
       
       # Plots from visualization:
       #+++++++++++++++++++++++++
       for key in plot_dict:
           current_fig = plot_dict[key][0]
           current_fig.savefig(self.output_loc + '/' + key+'.png')
           plt.close(current_fig)
       #+++++++++++++++++++++++++
       
       # Plots from post analysis:
       #+++++++++++++++++++++++++
       for key in ana_plots:
           current_fig = ana_plots[key][0]
           current_fig.savefig(self.output_loc + '/' + key+'.png')
           plt.close(current_fig)
       #+++++++++++++++++++++++++

       print("...done!")
       print("  ")     
       #**********************************
