class ConfigReader(object):
    
    # Initialize
    #*****************************
    def __init__(self,config_dict):
        self.config_dict = config_dict
    #*****************************

    # Read the config:
    #*****************************
    def load_setting(self,settings_name,default_value):
        setting = self.config_dict[settings_name] if settings_name in self.config_dict else default_value
        return setting
    #*****************************