from core.model_wrapper_core import ModelWrapper
from utils.torch_utils.torch_net_configurator import TorchNetConfigurator
from utils.torch_utils.torch_gradient_monitor import WeightGradientMonitor
from utils.config_reader import ConfigReader
from torchmetrics import MeanMetric, Accuracy
import torch
from torch import nn
import pickle

class AnupamNet(nn.Module,ModelWrapper):

    # Initialize:
    #*****************************
    def __init__(self,config,device):
        super().__init__()

        self.device = device
        self.config = config

        # Get the config reader to make our life a bit easer:
        self.config_reader = ConfigReader(config)

        # Use the configurator to get some base settings:
        self.net_configurator = TorchNetConfigurator()
        
        # Load model, if it is already trained
        self.model_load_path = self.config_reader.load_setting("model_load_path","")
        self.model_store_path = self.config_reader.load_setting("model_store_path","")
        self.model_name = self.config_reader.load_setting("model_name","anupam_net")

        # Get the model architecture, in case a new model is set up:
        self.n_inputs = self.config_reader.load_setting("n_inputs",-1)
        self.n_outputs = self.config_reader.load_setting("n_outputs",1)
        # Hidden layers:
        self.architecture = self.config_reader.load_setting("architecture",[100,100,100])
        self.n_layers = len(self.architecture)
        self.activations = self.config_reader.load_setting("activations",['Relu']*self.n_layers)
        self.weight_initialization = self.config_reader.load_setting("weight_initialization",['normal']*self.n_layers)
        self.bias_initialization = self.config_reader.load_setting("bias_initialization",['zeros']*self.n_layers)
        self.dropouts = self.config_reader.load_setting("dropouts",[0.0]*self.n_layers)
        self.batch_normalization = self.config_reader.load_setting("batch_normalization",[True]*self.n_layers)
        # output layer:
        self.output_activation = self.config_reader.load_setting("output_activation","softmax")
        self.output_weight_init = self.config_reader.load_setting("output_weight_init","normal")
        self.output_bias_init = self.config_reader.load_setting("output_bias_init","zeros")

        # Define the model optimizer(s), learning rate and learnig rate scheduler:
        # Jef:
        self.anupam_optimizer_str = self.config_reader.load_setting("anupam_optimizer","adam")
        self.anupam_learning_rate = self.config_reader.load_setting("anupam_learning_rate",1e-3)
        self.anupam_lr_scheduler_mode = self.config_reader.load_setting("anupam_lr_scheduler_mode","min")
        self.anupam_lr_scheduler_factor = self.config_reader.load_setting("anupam_lr_scheduler_factor",0.1)
        self.anupam_lr_scheduler_patience = self.config_reader.load_setting("anupam_lr_scheduler_patience",50)
        self.wrapped_model = None

        # Gradient monitoring, helpful to diagnose the model:
        self.watch_gradients = self.config_reader.load_setting("watch_gradients",True)

        # Handling outputs:
        self.show_model_response = self.config_reader.load_setting("show_model_response",False)

        # Get the loss function:
        self.loss_fn_str = self.config_reader.load_setting("loss_fn","categorical_crossentropy")
        # Load the anupam metrics in order to calculate the loss:
        # self.anupam_loss_fn = self.config_reader.load_setting("anupam_loss_fn",None)
        
        # assert self.anupam_loss_fn is not None,f">>> AnupamNet: ERROR. You need to provide a function for the anupam loss <<<"


        # Overwriteable parameters:
        # For now, we limit ourselves to the optimizer and learning rate, but
        # we might want to add more ...
        self.overwriteable_hp = ["optimizer","learning_rate"]

        # Training related parameters:
        self.seed = self.config_reader.load_setting("seed",None)
        if self.seed is not None:
          torch.manual_seed(self.seed)
          
        # BCE:
        # self.n_epochs_bce = self.config_reader.load_setting("n_epochs_bce",200)
        # self.batch_size_bce = self.config_reader.load_setting("batch_size_bce",256)
        # self.mon_epoch_bce = self.config_reader.load_setting("mon_epoch_bce",20)
        # self.read_epoch_bce = self.config_reader.load_setting("read_epoch_bce",1)
        # Jef:
        self.n_epochs_anupam = self.config_reader.load_setting("n_epochs_anupam",200)
        self.batch_size_anupam = self.config_reader.load_setting("batch_size_anupam",256)
        self.mon_epoch_anupam = self.config_reader.load_setting("mon_epoch_anupam",20)
        self.read_epoch_anupam = self.config_reader.load_setting("read_epoch_anupam",1)
    #*****************************

    # Handle model hyper parameters: (this will be important for the MLFlow integration and HPO)
    #*****************************
    # Set hyper parameters:
    def set_hyper_parameters(self,hp=None):
        if hp is not None:
           self.hp_dict = hp
        else:
           self.hp_dict = self.net_configurator.return_hp_dict()

    #------------------------------

    # Overwrite hyper parameters, in case we want to re-train the model with slightly different settings:
    def overwrite_hp(self,prefix='overwrite_'):
        #+++++++++++++++++++++
        for par in self.overwriteable_hp:
            key = prefix + par
            if key in self.config:
               self.hp_dict[par] = self.config[key]
        #+++++++++++++++++++++

    #------------------------------

    # Return hyper parameters:
    def return_hyper_parameters(self):
        return self.hp_dict
    #*****************************

    # Set up a fresh new model:
    #*****************************
    def setup_new_model(self):
        model_layers = self.net_configurator.get_dense_mlp_layers(
            n_inputs=self.n_inputs,
            n_outputs=self.n_outputs,
            architecture=self.architecture,
            activations=self.activations,
            weight_inits=self.weight_initialization,
            bias_inits=self.bias_initialization,
            dropouts=self.dropouts,
            batchnorms=self.batch_normalization,
            output_activation=self.output_activation,
            output_weight_init=self.output_weight_init,
            output_bias_init=self.output_bias_init
        )

        self.wrapped_model = nn.Sequential(model_layers).to(self.device)
    #*****************************

    # Build the model:
    #*****************************
    def build(self,model=None):
        # Use an existing model, if available:
        if model is not None:
          self.wrapped_model = model
          # Get the optimizer(s) and learning rate schedulers:
          # anupam:
          self.anupam_optimizer = self.net_configurator.set_optimizer(self.wrapped_model,self.anupam_optimizer_str,self.anupam_learning_rate)
          self.anupam_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.anupam_optimizer,mode=self.anupam_lr_scheduler_mode,factor=self.anupam_lr_scheduler_factor,patience=self.anupam_lr_scheduler_patience,verbose=True)
          # The optimizer for punzi training is set AFTER the BCE training stage
          # Set the loss-function
          self.loss_function = self.net_configurator.set_loss_function(self.loss_fn_str)
          # Need to turn off the reduction, in order to calculate the BCE lossL
          self.loss_function.reduction='none'

        # Check for an already trained model:
        if self.model_load_path != "" and self.model_name != "":
            # Load the model itself:
            full_model_path = self.model_load_path + '/' + self.model_name + ".pt"
            self.wrapped_model = torch.jit.load(full_model_path)
            self.wrapped_model.eval()

            # Load the hyper parameters, so that we can keep track and / or modify them:
            hp_dict_name = self.model_load_path + '/' + self.model_name + '_hp.pickle'
            with open(hp_dict_name, 'rb') as f:
               hp =  pickle.load(f)
               # Set hyper parameters:
               self.set_hyper_parameters(hp)

            # Overwrite parameters, if specified inside the configuration:
            self.overwrite_hp()
            # Get the optimizer(s) and learning rate schedulers:
            # # BCE:
            # self.bce_optimizer = self.net_configurator.set_optimizer(self.wrapped_model,self.bce_optimizer_str,self.bce_learning_rate)
            # self.bce_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.bce_optimizer,mode=self.bce_lr_scheduler_mode,factor=self.bce_lr_scheduler_factor,patience=self.bce_lr_scheduler_patience,verbose=True)
            # anupam:
            self.anupam_optimizer = self.net_configurator.set_optimizer(self.wrapped_model,self.anupam_optimizer_str,self.anupam_learning_rate)
            self.anupam_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.anupam_optimizer,mode=self.anupam_lr_scheduler_mode,factor=self.anupam_lr_scheduler_factor,patience=self.anupam_lr_scheduler_patience,verbose=True)
            # The optimizer for jef training is set AFTER the BCE training stage
            # Set the loss-function
            self.loss_function = self.net_configurator.set_loss_function(self.loss_fn_str)
            # Need to turn off the reduction, in order to calculate the BCE lossL
            self.loss_function.reduction='none'
        
        else:
            print("inside new model section")
            # Or build the model from scratch:
            self.setup_new_model()
            # Get the optimizer(s) and learning rate schedulers:
            # BCE:
            # self.bce_optimizer = self.net_configurator.set_optimizer(self.wrapped_model,self.bce_optimizer_str,self.bce_learning_rate)
            # self.bce_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.bce_optimizer,mode=self.bce_lr_scheduler_mode,factor=self.bce_lr_scheduler_factor,patience=self.bce_lr_scheduler_patience,verbose=True)
            # anupam:
            self.anupam_optimizer = self.net_configurator.set_optimizer(self.wrapped_model,self.anupam_optimizer_str,self.anupam_learning_rate)
            self.anupam_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.anupam_optimizer,mode=self.anupam_lr_scheduler_mode,factor=self.anupam_lr_scheduler_factor,patience=self.anupam_lr_scheduler_patience,verbose=True)            
            # Set the loss-function
            self.loss_function = self.net_configurator.set_loss_function(self.loss_fn_str)
            # Need to turn off the reduction, in order to calculate the BCE lossL
            self.loss_function.reduction='none'
            # And set the hyper parameters:
            self.set_hyper_parameters()

        # # Initialize trackers to monitor the performance:
        # # BCE:
        # self.train_bce_loss_tracker = MeanMetric().to(self.device)
        # self.val_bce_loss_tracker = MeanMetric().to(self.device)
        # self.train_bce_acc_tracker = Accuracy(task='binary').to(self.device)
        # self.val_bce_acc_tracker = Accuracy(task='binary').to(self.device)
        # Jef:
        self.train_anupam_loss_tracker = MeanMetric().to(self.device)
        self.val_anupam_loss_tracker = MeanMetric().to(self.device)
        self.train_anupam_acc_tracker = Accuracy(num_classes=5, task='multiclass').to(self.device)
        self.val_anupam_acc_tracker = Accuracy(num_classes=5, task='multiclass').to(self.device)

        # Load gradient monitors:
        # self.bce_grad_mon = WeightGradientMonitor(self.wrapped_model)
        self.anupam_grad_mon = WeightGradientMonitor(self.wrapped_model)
    #*****************************

    # Get the model response:
    #*****************************
    def forward(self,x):
        prediction = self.wrapped_model(x)
        prediction.to(self.device)
        return prediction
    #*****************************

    # Divide data into batches:
    #*****************************
    def get_data_batches(self,data_list,batch_dim):
        sample_size = data_list[0].size()[0]
        idx = None
        if batch_dim <= 0: # --> Use the entire data, but shuffle it:
          idx = torch.randint(low=0,high=sample_size,size=(sample_size,),device=self.device)
        else:
          idx = torch.randint(low=0,high=sample_size,size=(batch_dim,),device=self.device)  

        batched_data = []
        #++++++++++++++++
        for el in data_list:
            batched_data.append(el[idx].to(self.device))
        #++++++++++++++++
        return batched_data 
    #*****************************

    # Store the anupam-model:
    #*****************************
    def write_model_to_file(self,path):
        scripted_anupam = torch.jit.script(self.wrapped_model)
        scripted_anupam.save(path+'.pt')

        # code for writing in ONNX format
        # Create a dummy input with the shape that your model expects
        input_shape = (self.n_inputs,)
        dummy_input = torch.randn(1, *input_shape).to(self.device)  # replace `input_shape` with actual input dimensions

        onnx_path = path + ".onnx"
        print(f"Exporting model to ONNX format at {onnx_path}...")

        # Export to ONNX
        torch.onnx.export(
            self.wrapped_model,         # Model to export
            dummy_input,                # Dummy input tensor
            onnx_path,                  # Output file path for ONNX
            export_params=True,         # Store the trained parameters
            opset_version=11,           # ONNX version
            do_constant_folding=True,   # Optimize constant expressions
            input_names=["input"],      # Input node name
            output_names=["output"],    # Output node name
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}  # Support variable batch sizes
        )

        print(f"Model successfully exported to ONNX format at {onnx_path}")
    #*****************************

    # Compute the jef objective:
    def get_anupam_objective(self,model_prediction,y,w):
        return self.loss_function(model_prediction,y)*w  / torch.sum(w)
    #-------------------------------

    # anupam training:
    def anupam_train_step(self,x,y,w):
        # Reset the optimizer:
        self.anupam_optimizer.zero_grad(set_to_none=True)
        # Get the model predictions and compute the loss:
        model_predictions = torch.squeeze(self.forward(x))
        y = torch.squeeze(y).long()
        anupam_loss = torch.mean(self.get_anupam_objective(model_predictions,y,w))
        anupam_loss.backward()
        # Update weights:
        self.anupam_optimizer.step()

        # Record the loss:
        self.train_anupam_loss_tracker.update(anupam_loss)
        predicted_classes = torch.argmax(model_predictions, dim=1)
        
        # Record the accuracy:
        self.train_anupam_acc_tracker(predicted_classes,y)
    #-------------------------------

    # anupam testing:
    def anupam_test_step(self,x,y,w):   
        # Get the model predictions and compute the loss:
        model_predictions = torch.squeeze(self.forward(x))
        y = torch.squeeze(y).long()
        anupam_loss = torch.mean(self.get_anupam_objective(model_predictions,y,w))
        # Record the loss:
        self.val_anupam_loss_tracker.update(anupam_loss)
        predicted_classes = torch.argmax(model_predictions, dim=1)
        # Record the accuracy:
        self.val_anupam_acc_tracker(predicted_classes,y)

    #-------------------------------

    # Now combine everythin into one fit function:  
    def fit(self,x,y,w,s=None,x_val=None,y_val=None,w_val=None,s_val=None):

        anupam_training_loss = []
        anupam_validation_loss = []
        anupam_training_acc = []
        anupam_validation_acc = []

        run_testing = False
        if x_val is not None and y_val is not None and w_val is not None:
            run_testing = True
    
        
        if self.n_epochs_anupam > 0:
            #+++++++++++++++++++++++++++++++
            for epoch in range(1,1+self.n_epochs_anupam):
                # Divide data into batches:
                x_train, y_train, w_train = self.get_data_batches([x,y,w],self.batch_size_anupam)
                # Update the nework:
                self.anupam_train_step(x_train,y_train, w_train)

                # Monitor gradients, if requested:
                if self.watch_gradients:
                  self.anupam_grad_mon.watch_gradients_per_batch(self.read_epoch_anupam)

                # Run testing, if data is provided:
                if run_testing:
                    x_test, y_test, w_test = self.get_data_batches([x_val,y_val,w_val],self.batch_size_anupam)

                    # Test everything:
                    self.anupam_test_step(x_test,y_test, w_test)

                # Record the losses:
                if epoch % self.read_epoch_anupam == 0:
                    anupam_training_loss.append(self.train_anupam_loss_tracker.compute().detach().cpu().item())
                    anupam_training_acc.append(self.train_anupam_acc_tracker.compute().detach().cpu().item())
                    
                    self.train_anupam_loss_tracker.reset()
                    self.train_anupam_acc_tracker.reset()

                    # Collect the gradients:
                    if self.watch_gradients:
                        self.anupam_grad_mon.collect_gradients_per_epoch()
                
                    if run_testing:
                        anupam_validation_loss.append(self.val_anupam_loss_tracker.compute().detach().cpu().item())
                        anupam_validation_acc.append(self.val_anupam_acc_tracker.compute().detach().cpu().item())

                        self.val_anupam_loss_tracker.reset()
                        self.val_anupam_acc_tracker.reset()

                # Print out some info:
                if epoch % self.mon_epoch_anupam == 0:
                    if len(anupam_training_loss) > 0:
                        print(" ")
                        print("Anupam epoch: " + str(epoch) + "/" + str(self.n_epochs_anupam))
                        print("Anupam loss: " + str(round(anupam_training_loss[-1],8)))
                        print("Anupam accuracy: " + str(round(anupam_training_acc[-1],3)))
                        print("Anupam validation loss: " + str(round(anupam_validation_loss[-1],8)))
                        print("Anupam validation accuracy: " + str(round(anupam_validation_acc[-1],3)))

                    # if len(bce_validation_loss) > 0:
                    #     print("Anupam validation loss: " + str(round(jef_validation_loss[-1],3)))
                    #     print("Anupam validation accuracy: " + str(round(jef_validation_acc[-1],3)))
            #+++++++++++++++++++++++++++++++
            print(" ")

            # Store the jef net after Jef-training, if a path for storage is provided:
            if self.model_store_path != "" and self.model_store_path is not None:
                print("model_store_path",self.model_store_path)
                self.write_model_to_file(self.model_store_path+'/'+self.model_name+'_post_anupam')    

                # Write the gradients (if available) to the same path where the model is stored:
                if self.watch_gradients:
                    anupam_gradients = self.anupam_grad_mon.read_out_gradients()
                    anupam_gradient_plots = self.anupam_grad_mon.show_gradients(
                        gradient_dict = anupam_gradients,
                        model_name = self.model_name+'_post_anupam',
                        xlabel='Epoch per ' + str(self.read_epoch_anupam)
                    )
                    
                    # Write gradients to file:
                    #+++++++++++++++++++
                    for el in anupam_gradient_plots:
                        current_fig = anupam_gradient_plots[el][0]
                        current_fig.savefig(self.model_store_path+'/'+el+'.png')
                    #+++++++++++++++++++
        else:
            print(">>> AnupamNet: INFO. Anupam training has been disabled. <<<")

        return {
            'anupam_loss': anupam_training_loss,
            'anupam_val_loss': anupam_validation_loss,        
            'anupam_acc': anupam_training_acc,
            'anupam_val_acc': anupam_validation_acc,
            }
    #*****************************

    #*****************************
    def predict(self,x):
        return self.forward(x)
    #*****************************

    
