import torch
from torch import nn
import math
from collections import OrderedDict

class TorchNetConfigurator(object):

    # Initialize:
    #**************************
    def __init__(self):

        # For bookkeeping: store all hyper parameters:
        self.hp_dict = {}

        # Default layer parameters:
        # Dropout layer:
        self.dropout_inplace = False
        # Linear layer:
        self.linearL_bias = True
        self.linearL_device = None
        self.linearL_dtype = None

        # Batch normalization:
        self.batchnormL_eps = 1e-5
        self.batchnormL_momentum = 0.1
        self.batchnormL_affine = True,
        self.batchnormL_track_running_stats=True
        self.batchnormL_device = None
        self.batchnormL_dtype = None
        
        # Parameters for various activation functions: (these are the default values from pytorch)
        self.elu_alpha = 1.0
        self.leaky_relu_slope = 0.01
        self.selu_inplace = False
        # Please feel free to add more parameters for more activation functions...

        # Parameters for various optimizers: (these are the default values from pytorch)
        # Adam: (for details please see: https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam)
        self.adam_betas = (0.9, 0.999)
        self.adam_eps =  1e-08
        self.adam_weight_decay = 0
        self.adam_amsgrad = False
        self.adam_foreach = None
        self.adam_maximize = False
        self.adam_capturable = False
        self.adam_differentiable = False
        self.adam_fused = None

        # SGD: (details can be found here: https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD)
        self.sgd_momentum = 0
        self.sgd_dampening = 0
        self.sgd_weight_decay = 0
        self.sgd_nesterov = False
        self.sgd_maximize = False
        self.sgd_foreach = None
        self.sgd_differentiable = False

        # RMSprop (details can be found here: https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html#torch.optim.RMSprop)
        self.rmsprop_alpha = 0.99
        self.rmsprop_eps = 1e-08
        self.rmsprop_weight_decay = 0
        self.rmsprop_momentum = 0
        self.rmsprop_centered = False
        self.rmsprop_maximize = False
        self.rmsprop_foreach = None
        self.rmsprop_differentiable = False

        # Please feel free to add more optimizers...
    #**************************

    # Set the activation functions:
    #**************************
    def set_activation_function(self,act_func_str):
        if act_func_str.lower() == "relu":
            return nn.ReLU()
        
        if act_func_str.lower() == "leaky_relu":
            return nn.LeakyReLU(self.leaky_relu_slope)
        
        if act_func_str.lower() == "elu":
            return nn.ELU(self.elu_alpha)
        
        if act_func_str.lower() == "selu":
            return nn.SELU(self.selu_inplace)
        
        if act_func_str.lower() == "tanh":
            return nn.Tanh()
        
        if act_func_str.lower() == "sigmoid":
            return nn.Sigmoid()
        
        if act_func_str.lower() == "softmax":
            return nn.Softmax()
        
        # If no activation is provided or set to 'linear', then return -1:
        if act_func_str.lower() == "linear" or act_func_str == "" or act_func_str is None:
            return -1
        
        # Add more activations (see here: https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity)
    #**************************

    # Get the loss function:
    #**************************
    def set_loss_function(self,loss_fn_str):
        # Add info to the hyper parameter dict:
        self.hp_dict['loss_fn'] = loss_fn_str

        if loss_fn_str.lower() == 'mse':
            return torch.nn.MSELoss()
        elif loss_fn_str.lower() == 'mae':
            return torch.nn.L1Loss()
        elif loss_fn_str.lower() == 'huber':
            return torch.nn.HuberLoss()
        elif loss_fn_str.lower() == 'categorical_crossentropy':
            return torch.nn.CrossEntropyLoss()
        elif loss_fn_str.lower() == 'binary_crossentropy':
            return torch.nn.BCELoss()
        else:
            self.hp_dict['loss_fn'] = "mse"
            print(">>> TorchNetConfigurator: WARNING! This loss function is (currently) not implemented. Going to use default: MSE <<<")
            return torch.nn.MSELoss()
        
        # Add more losses (see here: https://pytorch.org/docs/stable/nn.html#loss-functions)
    #**************************

    # Set the optimizer:
    #**************************
    def set_optimizer(self,model,optimizer_name,learning_rate):
        # Add info to the hyper parameter dict:
        self.hp_dict['optimizer'] = optimizer_name
        self.hp_dict['learning_rate'] = learning_rate

        if optimizer_name.lower() == 'adam':
            return torch.optim.Adam(
                params=model.parameters(),
                lr=learning_rate,
                betas=self.adam_betas,
                eps=self.adam_eps,
                weight_decay=self.adam_weight_decay,
                amsgrad=self.adam_amsgrad,
                foreach=self.adam_foreach,
                maximize=self.adam_maximize,
                capturable=self.adam_capturable,
                differentiable=self.adam_differentiable,
                fused=self.adam_fused
                )
        
        elif optimizer_name.lower() == 'sgd':
            return torch.optim.SGD(
                params=model.parameters(),
                lr=learning_rate,
                momentum=self.sgd_momentum,
                dampening=self.sgd_dampening,
                weight_decay=self.sgd_weight_decay,
                nesterov=self.sgd_nesterov,
                maximize=self.sgd_maximize,
                foreach=self.sgd_foreach,
                differentiable=self.sgd_differentiable
                )
        
        elif optimizer_name.lower() == 'rmsprop':
            return torch.optim.RMSprop(
                params=model.parameters(),
                lr=learning_rate,
                alpha=self.rmsprop_alpha,
                eps=self.rmsprop_eps,
                weight_decay=self.rmsprop_weight_decay,
                momentum=self.rmsprop_momentum,
                centered=self.rmsprop_centered,
                maximize=self.rmsprop_maximize,
                foreach=self.rmsprop_foreach,
                differentiable=self.rmsprop_differentiable
            )
        
        else:
            self.hp_dict['optimizer'] = "adam"

            print(">>> TorchNetConfigurator: WARNING! This optimizer is (currently) not implemented. Going to use default: Adam <<<")
            return torch.optim.Adam(
                params=model.parameters(),
                lr=learning_rate,
                betas=self.adam_betas,
                eps=self.adam_eps,
                weight_deacy=self.adam_weight_decay,
                amsgrad=self.adam_amsgrad,
                foreach=self.adam_foreach,
                maximize=self.adam_maximize,
                capturable=self.adam_capturable,
                differentiable=self.adam_differentiable,
                fused=self.adam_fused
                )
    #**************************

    # Set the weight initialization:
    # This is quite important!!!
    #************************** 
    def initialize_linear_layer(self,layer,layer_activation,weight_init,bias_init):
        # Get the weights and bias first:
        w = None
        b = None

        if layer.weight is not None:
           w = layer.weight.data
        if layer.bias is not None:
           b = layer.bias.data

        # Handle weight initialization:
        if weight_init.lower() != "default" and w is not None: #--> Default here means the default pytorch implementation...
           if layer_activation.lower == 'linear' or layer_activation.lower() == 'tanh' or layer_activation.lower() == 'sigmoid' or layer_activation.lower() == 'softmax':
               if weight_init.lower() == 'normal':
                   torch.nn.init.xavier_normal_(w)
               if weight_init.lower() == 'uniform':
                   torch.nn.init.xavier_uniform_(w)

           if layer_activation.lower() == 'relu' or layer_activation.lower() == 'leaky_relu' or layer_activation.lower() == 'elu':
               a_slope = 0.0
               if layer_activation.lower() == 'leaky_relu':
                   a_slope = self.leaky_relu_slope

               if weight_init.lower() == 'normal':
                  torch.nn.init.kaiming_normal_(w,a=a_slope,nonlinearity=layer_activation.lower())
               if weight_init.lower() == 'uniform':
                  torch.nn.init.kaiming_uniform_(w,a=a_slope,nonlinearity=layer_activation.lower())
          
           # Add the lecun initialization for selu activation:
           # Following this definition: https://www.tensorflow.org/api_docs/python/tf/keras/initializers/LecunNormal 
           if layer_activation.lower() == 'selu':
              stddev = 1. / math.sqrt(w.size(1))
              torch.nn.init.normal_(w,mean=0.0,std=stddev)

        # Handle bias initialization: #--> Default here means the default pytorch implementation...
        if bias_init.lower() != "default" and b is not None:
            if bias_init.lower() == "normal":
                torch.nn.init.normal_(b)
            if bias_init.lower() == "uniform":
                torch.nn.init.uniform_(b)
            if bias_init.lower() == "ones":
                torch.nn.init.ones_(b)
            if bias_init.lower() == "zeros":
                torch.nn.init.zeros_(b) 

        # Add more initialization methods...
    #**************************

    # Set up the layers for a dense mlp:
    #**************************
    def get_dense_mlp_layers(self,n_inputs,n_outputs,architecture,activations,weight_inits,bias_inits,dropouts,batchnorms,output_activation,output_weight_init,output_bias_init):
        # Get the number of layers:
        n_layers = len(architecture)
        # First, make sure that the dimensionality is correct:
        assert n_inputs > 0, f">>> TorchNetConfigurator: ERROR! Number of inputs {n_inputs} has to be positive <<<"
        assert n_outputs > 0, f">>> TorchNetConfigurator: ERROR! Number of outputs {n_outputs} has to be positive <<<"
        assert n_layers == len(activations), f">>> TorchNetConfigurator: ERROR! Number of hidden layers {len(architecture)} does not match the number of activations {len(activations)} <<<"
        assert n_layers == len(dropouts), f">>> TorchNetConfigurator: ERROR! Number of hidden layers {len(architecture)} does not match the number of dropout values {len(dropouts)} <<<"
        assert n_layers == len(batchnorms), f">>> TorchNetConfigurator: ERROR! Number of hidden layers {len(architecture)} does not match the number of batchnorm values {len(batchnorms)} <<<"
        assert n_layers == len(weight_inits), f">>> TorchNetConfigurator: ERROR! Number of hidden layers {len(architecture)} does not match the number of weight initializations {len(weight_inits)} <<<"
        assert n_layers == len(bias_inits), f">>> TorchNetConfigurator: ERROR! Number of hidden layers {len(architecture)} does not match the number of bias initializations {len(bias_inits)} <<<"

        # Add info to the hyper parameter dict:
        self.hp_dict['n_inputs'] = n_inputs
        self.hp_dict['n_outputs'] = n_outputs
 
        # Now we can set up the mlp:
        mlp_layers = OrderedDict()
        
        # Take care of the hidden units:
        n_prev_nodes = n_inputs
        #++++++++++++++++++++++++++
        for i in range(n_layers):
            layer_name = 'layer_' + str(i)
            act_name = 'activation_' + str(i)
            dropout_name = 'dropout_' + str(i)
            batchnorm_name = 'batchnorm_' + str(i)
            
            # Add some neurons
            mlp_layers[layer_name] = nn.Linear(
                in_features=n_prev_nodes,
                out_features=architecture[i],
                bias=self.linearL_bias,
                device=self.linearL_device,
                dtype=self.linearL_dtype
            )
            self.hp_dict[layer_name+'_units'] = architecture[i]

            # Set the activation function:
            layer_activation = self.set_activation_function(activations[i])
            if layer_activation != -1:
                mlp_layers[act_name] = layer_activation
            self.hp_dict[act_name] = activations[i]

            # Now initialize the layer properly:
            self.initialize_linear_layer(mlp_layers[layer_name],activations[i],weight_inits[i],bias_inits[i])
            
            # Add a batch normalization (if requested):
            if batchnorms[i] == True:
                mlp_layers[batchnorm_name] = nn.BatchNorm1d(
                    num_features = architecture[i],
                    eps=self.batchnormL_eps,
                    momentum=self.batchnormL_momentum,
                    affine=self.batchnormL_affine,
                    track_running_stats=self.batchnormL_track_running_stats,
                    device=self.batchnormL_device,
                    dtype=self.batchnormL_dtype
                )
            self.hp_dict[batchnorm_name] = batchnorms[i]
            
            # Include a dropout, if requested:
            if dropouts[i] > 0.0:
                mlp_layers[dropout_name] = nn.Dropout(p=dropouts[i],inplace=self.dropout_inplace)
            self.hp_dict[dropout_name] = dropouts[i]
            
            n_prev_nodes = architecture[i]
        #++++++++++++++++++++++++++
        
        # Add an output:
        mlp_layers['output_layer'] = nn.Linear(n_prev_nodes,n_outputs)
        output_act = self.set_activation_function(output_activation)
        if output_act != -1:
            mlp_layers['output_activation'] = output_act
        self.hp_dict['output_activation'] = output_activation


        # Initialize the output:
        self.initialize_linear_layer(mlp_layers['output_layer'],output_activation,output_weight_init,output_bias_init)

        # And return it:
        return mlp_layers
    #**************************

    # Make the hyper parameters available:
    #**************************
    def return_hp_dict(self):
        if not self.hp_dict:
            print(">>> TorchNetConfigurator: WARNING! No hyper parameters have been set. Going to return an empty dictionary <<<")
        
        return self.hp_dict
    #**************************

    