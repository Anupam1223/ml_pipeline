import torch
from torchmetrics import Metric
import numpy as np
import matplotlib.pyplot as plt

class WeightGradientMonitor(object):

    """
    This class is used to monitor the gradient flow of a neural network

    Important Notes:
    ---------------- 
    
    (i) Only the gradients of the layer weights are monitored 

    (ii) Parts of the code used here where inspired by / taken from this discussion on the pytorch forum:
         https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/7

    (iii) Here, three quantities are calculated for each layer in the network:
          1. The average abosulte gradient over all weights
          2. The maximum absolute gradient of all weights
          3. The minimum absolute gradient of all weights

    (iv) The gradients are checked for each batch sample. In order to keep the accumulated data size low, (we do not want to keep track of all gradients in every single batch)
    the weighted mean of each quantity is calculated over all batch samples. This means: Per training epoch, the user may access the weighted mean of:
    1. the average absolute gradient, 2. the maximum absolute gradient and 3. the minimum absolute gradient.

    (v) The implementation of this tool can be seen / tested in the demo folder under: gradient_demo/run_grad_mon_demo.py

    (vi) We use the pytorch metrics tool (see website here: https://torchmetrics.readthedocs.io/en/stable/pages/implement.html) to formualte our 
    metric class that calculates the weighted mean. This way, we can ensure that the graient information is processed in a consistent and efficient way.
    """

    # Initialize: 
    #******************************************
    def __init__(self,model,excluded_layer_idx=[-1],config={}):
        self.model = model
        self.excluded_layer_idx = excluded_layer_idx #--> Here, the user might add the indices of layers that shall not be considered
        # during the gradient analysis, e.g. a custom layer with weird parameters. Lets say the second and fourth hidden layer of a model shall be excluded,
        # then one would simply set: exclude_layer_idx = [2,4]

        # Run a small evaluation of the model:
        self.active_layer_idx = [] #--> Needed for later, when the results are visualized
        layer_counter =  0
        acc_layer_counter = 0
        #++++++++++++++++++++++++++++++
        for name, weights in self.model.named_parameters():
            if "bias" not in name: #--> We are not interested in the bias values here
                layer_counter += 1

                if layer_counter not in self.excluded_layer_idx: #--> We do not count excluded layers
                    acc_layer_counter += 1

                    if weights.requires_grad: #--> Look at active neurons / layers only
                       self.active_layer_idx.append(acc_layer_counter)
        #++++++++++++++++++++++++++++++

        self.n_layers = acc_layer_counter
        self.n_active_layers = len(self.active_layer_idx)

        self.average_gradient_watched = GradientTracker(self.n_active_layers)
        self.minimum_gradient_watched = GradientTracker(self.n_active_layers)
        self.maximum_gradient_watched = GradientTracker(self.n_active_layers)
    
        # Once all batches are analyzed, we collect the gradients
        self.average_gradient_collected = []
        self.minimum_gradient_collected = []
        self.maximum_gradient_collected = []

        # Settings for plotting the results:
        self.config = config
        self.line_width = self.config['line_width'] if 'line_width' in self.config else 2.0
        self.fig_width_scale = self.config['fig_width_scale'] if 'fig_width_scale' in self.config else 1.0
        self.fig_height_scale = self.config['fig_height_scale'] if 'fig_height_scale' in self.config else 3.0
        self.legend_font_size = self.config['legend_font_size'] if 'legend_font_size' in self.config else 15
    #******************************************

    # Reset everything:
    #******************************************
    # Watched gradients
    def reset_watch(self):
        self.average_gradient_watched.reset()
        self.maximum_gradient_watched.reset()
        self.minimum_gradient_watched.reset()

    #-----------------------

    # Collected gradients:
    def reset_collection(self):
        self.average_gradient_collected = []
        self.minimum_gradient_collected = []
        self.maximum_gradient_collected = []
    #******************************************

    # Monitor the gradients:
    # This function is called per batch
    #******************************************
    def watch_gradients_per_batch(self,sample_size):
        avg_grad = torch.zeros(self.n_active_layers)
        min_grad = torch.zeros(self.n_active_layers)
        max_grad = torch.zeros(self.n_active_layers)

        norm = torch.tensor((1.0 / float(sample_size)))
    
        i = 0
        #++++++++++++++++++++++++
        for name, weights in self.model.named_parameters():
            if(weights.requires_grad) and ("bias" not in name):
                current_gradient = weights.grad.abs().cpu()

                avg_grad[i] = torch.mean(current_gradient)
                max_grad[i] = torch.max(current_gradient)
                min_grad[i] = torch.min(current_gradient)

                i += 1
        #++++++++++++++++++++++++

        self.average_gradient_watched.update(avg_grad,norm)
        self.maximum_gradient_watched.update(max_grad,norm)
        self.minimum_gradient_watched.update(min_grad,norm)
    #******************************************

    # Collect the gradients:
    # This function is called once per epoch:
    #******************************************
    def collect_gradients_per_epoch(self):
        self.average_gradient_collected.append(self.average_gradient_watched.compute().numpy())
        self.minimum_gradient_collected.append(self.minimum_gradient_watched.compute().numpy())
        self.maximum_gradient_collected.append(self.maximum_gradient_watched.compute().numpy())

        # Now reset everything, because we start a new watch at the next training epoch
        self.reset_watch()
    #******************************************
    
    # Visualize the gradients:
    #******************************************
    # Read them out:
    def read_out_gradients(self):
        average_gradient_np = np.array(self.average_gradient_collected)
        minimum_gradient_np = np.array(self.minimum_gradient_collected)
        maximum_gradient_np = np.array(self.maximum_gradient_collected)
        
        # Now we clean up everything
        self.reset_watch()
        self.reset_collection()

        return {
            'average_gradients': average_gradient_np,
            'maximum_gradients': maximum_gradient_np,
            'minimum_gradients': minimum_gradient_np
        }

    #-------------------------------------------
    
    # Plot the gradients:
    # This function does not plot the results right away. The user needs to call
    # 'plt.show()' or: 'fig.savefig()' after this function, in order to show / store the results
    def show_gradients(self,gradient_dict,model_name,xlabel='Epoch'):
        average_gradients = gradient_dict['average_gradients']
        maximum_gradients = gradient_dict['maximum_gradients']
        minimum_gradients = gradient_dict['minimum_gradients']  

        fig_height = int(self.fig_height_scale * self.n_active_layers)
        fig_width = int(self.fig_width_scale * 15)

        fig_size=(fig_width,fig_height)
        fig,ax = plt.subplots(self.n_active_layers,1,sharex=True,figsize=fig_size)
        fig.suptitle(model_name + ' Gradient Flow')

        #++++++++++++++++++++++
        for k in range(self.n_active_layers):
            layer_name = 'Hidden Layer ' + str(self.active_layer_idx[k])

            if self.active_layer_idx[k] == self.n_layers:
                layer_name = 'Output Layer'

            ax[k].plot(minimum_gradients[:,k],'r-',linewidth=self.line_width,label='Min.')
            ax[k].plot(maximum_gradients[:,k],'g--',linewidth=self.line_width,label='Max.')
            ax[k].plot(average_gradients[:,k],'k-',linewidth=self.line_width,label='Avg.')

            ax[k].set_ylabel(layer_name)
            ax[k].legend(fontsize=self.legend_font_size)
            ax[k].grid(True)
        #++++++++++++++++++++++
        ax[self.n_active_layers-1].set_xlabel(xlabel)
        plt.close()

        key_name = 'gradient_flow_' + model_name
        return {
          key_name: [fig,ax] #--> The pyplot figure and axis are stored, so that any user has the option to make some changes (e.g. the axis labeling or figure size)
        }
    #******************************************


class GradientTracker(Metric):

    # Initialize:
    #****************************
    def __init__(self,n_layers):
        super().__init__()
        self.add_state("gradient_tensor", default=torch.zeros(n_layers),dist_reduce_fx="sum")
        self.add_state("sum_weights",default=torch.tensor(0.0),dist_reduce_fx="sum")
    #****************************

    # Update the state:
    #****************************
    def update(self,gradient_values : torch.Tensor, weight : torch.Tensor):
        self.gradient_tensor += gradient_values * weight
        self.sum_weights += weight
    #****************************

    # Get the results:
    #****************************
    def compute(self):
        avg_mean = self.gradient_tensor / self.sum_weights
        return avg_mean
    #****************************

    
