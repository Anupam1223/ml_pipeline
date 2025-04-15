from mlflow_jlab.core.visualization_core import Visualization
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from sklearn import metrics
from mlflow_jlab.utils.config_reader import ConfigReader
import mlflow_jlab.model_wrapper as wrappers


class AnupamVisualizer(Visualization):

    # Initialize:
    #**********************************
    def __init__(self,config):
        self.config=config

        # Get the config reader to make our life a bit easer:
        self.config_reader = ConfigReader(config)

        # Decide to turn off the ROC curve / threshold scan:
        self.disable_roc_curve = self.config_reader.load_setting("disable_roc_curve",False)
        self.disable_threshold_scan = self.config_reader.load_setting("disable_threshold_scan",False)

        # Some basic plotting definitions:
        self.font_size = self.config_reader.load_setting("font_size",20)
        self.leg_font_size = self.config_reader.load_setting("leg_font_size",15)

        # Loss and accuracy plots:
        self.loss_plot_figsize = self.config_reader.load_setting("loss_plot_figsize",(22,8))
        self.loss_plot_wspace = self.config_reader.load_setting("loss_plot_wspace",0.7)
        self.loss_plot_name = self.config_reader.load_setting("loss_plot_name","jef_loss_and_accuracy")

        # Jef response:
        self.response_plot_figsize = self.config_reader.load_setting("response_plot_figsize",(18,8))
        self.response_plot_nbins = self.config_reader.load_setting("response_plot_nbins",100)
        self.response_plot_linewidth = self.config_reader.load_setting("response_plot_linewidth",3.0)
        self.response_plot_name = self.config_reader.load_setting("response_plot_name","jef_response")
        self.response_plot_log = self.config_reader.load_setting("response_plot_log",True)

        # Jef ROC:
        self.roc_figsize = self.config_reader.load_setting("roc_figsize",(12,8))
        self.roc_linewidth = self.config_reader.load_setting("roc_linewidth",3.0)
        self.roc_name = self.config_reader.load_setting("roc_name","jef_ROC_curves")

        # Threshold scan:
        self.n_th_scans = self.config_reader.load_setting("n_th_scans",10)
        self.th_scan_start = self.config_reader.load_setting("th_scan_start",0.0)
        self.th_scan_figsize = self.config_reader.load_setting("th_scan_figsize",(12,8))
        self.th_scan_linewidth = self.config_reader.load_setting("th_scan_linewidth",3.0)
        self.th_scan_markersize = self.config_reader.load_setting("th_scan_markersize",10.0)
        self.th_scan_name = self.config_reader.load_setting("th_scan_name","jef_threshold_scan")


        plt.rcParams.update({"font.size":self.font_size})
    #**********************************
    
    # Plot the results:
    #**********************************
    def visualize(self,loss_dictionary,bce_loss_x_label,anupam_loss_x_label,data_labels=None):

        plot_dict = {}

        # PLot the loss and accuracy of the anupam Net:
        figl,axl = plt.subplots(1,4,figsize=self.loss_plot_figsize)
        figl.subplots_adjust(wspace=self.loss_plot_wspace)

        axl[2].plot(loss_dictionary['anupam_loss'],linewidth=3.0,label='Training')
        axl[2].plot(loss_dictionary['anupam_val_loss'],linewidth=3.0,label='Validation')
        axl[2].set_xlabel(anupam_loss_x_label)
        axl[2].set_ylabel('Anupam Loss')
        axl[2].legend(fontsize=self.leg_font_size)
        axl[2].grid(True)

        axl[3].plot(loss_dictionary['anupam_acc'],linewidth=3.0,label='Training')
        axl[3].plot(loss_dictionary['anupam_val_acc'],linewidth=3.0,label='Validation')
        axl[3].set_xlabel(bce_loss_x_label)
        axl[3].set_ylabel('Anupam Accuracy')
        axl[3].legend(fontsize=self.leg_font_size)
        axl[3].grid(True)

        plot_dict[self.loss_plot_name] = [figl,axl]
        plt.close(figl)

        return plot_dict
    #**********************************


