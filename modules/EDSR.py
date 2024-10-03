import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pickle


class ResBlock(nn.Module):
    
    def __init__(self, channels):
        """ Residual block for EDSR model 
        
        Parameters
        ----------
        channels : int
            Number of channels in the input tensor
        """
        super(ResBlock, self).__init__()
        self.channels = channels
        self.conv1 = nn.Conv2d(in_channels=channels, 
                               out_channels=channels, 
                               kernel_size=3,
                               padding=1,
                               bias=False)
        self.conv2 = nn.Conv2d(in_channels=channels, 
                               out_channels=channels, 
                               kernel_size=3,
                               padding=1,
                               bias=False)
    def forward(self, x):
        res = self.conv1(x)
        res = F.relu(res)
        res = self.conv2(res)
        return torch.add(x, res)
    


class UpSample(nn.Module):
    
    def __init__(self, channels, factor):
        """ Upsample block for EDSR model

        Parameters
        ----------
        channels : int
            Number of channels in the input tensor
        factor : int
            Upsampling factor
        """
        super(UpSample, self).__init__()
        self.channels = channels
        self.factor = factor
        self.conv1 = nn.Conv2d(in_channels=channels, 
                               out_channels=int(channels * factor**2), 
                               kernel_size=3,
                               padding=1,
                               bias=False)
        self.conv2 = nn.Conv2d(in_channels=channels, 
                               out_channels=int(channels * factor**2), 
                               kernel_size=3,
                               padding=1,
                               bias=False)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.pixel_shuffle(x, upscale_factor=self.factor)
        x = self.conv2(x)
        x = F.pixel_shuffle(x, upscale_factor=self.factor)
        return x
    


class EDSRModel(nn.Module):
    
    def __init__(self, channels, num_resblocks, factor):
        """ EDSR model for super-resolution

        Parameters
        ----------
        channels : int
            Number of channels in the input tensor
        num_resblocks : int
            Number of residual blocks
        factor : int
            Upsampling factor
        """
        super(EDSRModel, self).__init__()
        self.channels = channels
        self.num_resblocks = num_resblocks
        self.factor = factor
        
        self.conv1 = nn.Conv2d(in_channels=3, 
                               out_channels=channels, 
                               kernel_size=3,
                               padding=1,
                               bias=False)
        
        self.res_stack = nn.ModuleList()
        for _ in range(num_resblocks):
            self.res_stack.append(ResBlock(channels))
        
        self.conv2 = nn.Conv2d(in_channels=channels, 
                               out_channels=channels, 
                               kernel_size=3,
                               padding=1,
                               bias=False)
        
        self.upsample = UpSample(channels=channels, 
                                 factor=factor)
        
        self.conv3 = nn.Conv2d(in_channels=channels, 
                               out_channels=3, 
                               kernel_size=3,
                               padding=1,
                               bias=False)
        
        
    def forward(self, inputs):
        x = self.conv1(inputs)
        res = x
        for i in range(self.num_resblocks):
            res = self.res_stack[i](res)
        res = self.conv2(res)
        x = torch.add(x, res)
        x = self.upsample(x)
        out = self.conv3(x)
        return out
    

    
    def save_model(self, save_folder):
        "Save the parameters and the model state_dict"
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
            
        param_file = os.path.join(save_folder, 'parameters.pkl')
        parameters = [self.channels,
                      self.num_resblocks,
                      self.factor]
        with open(param_file, 'wb') as f:
            pickle.dump(parameters, f)
        
        model_file = os.path.join(save_folder, 'model.pt')
        torch.save(self.state_dict(), model_file)


    def save_history(self, history, save_folder):
        "Save the training history"
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        
        history_file = os.path.join(save_folder, 'history.pkl')
        with open(history_file, 'wb') as f:
            pickle.dump(history, f)
    

    @classmethod
    def load_model(cls, save_folder):
        "Load the model from a folder"
        param_file = os.path.join(save_folder, 'parameters.pkl') 
        with open(param_file, 'rb') as f:
            parameters = pickle.load(f)
                
        model = cls(*parameters)
            
        model_file = os.path.join(save_folder, 'model.pt')
        model.load_state_dict(torch.load(model_file, map_location='cuda:0'))
            
        return model
    

    @classmethod
    def load_history(cls, save_folder):
        "Load the training history from a folder"
        history_file = os.path.join(save_folder, 'history.pkl')
        with open(history_file, 'rb') as f:
            history = pickle.load(f)
        
        return history
        
        

    
        
        
        
    
        

