from embiggen import * 
from utils import * 
from torch.utils.data import Dataset
import torch.nn.functional as F
from PIL import Image

class ProBaVdataset(Dataset):
    def __init__(self, data_path, mode = 'train', num_top_clearance = 9, transform = False):
        super(ProBaVdataset, self).__init__()
        self.transfrom = transform
        self.mode = mode
        self.num_input = 9
        self.data_path = data_path
    def __getitem__(self, index):
        if self.mode == 'train':
            lr, sm = get_lr_sm_from_scene(self.data_path[index])
            hr =  highres_image(self.data_path[index])[0]
            sm = np.squeeze(sm, axis=1)
            #select n lr images based on degree of clearance 
            sorted_by_clearance = np.sum(sm, axis = (1,2)).argsort()[-9:][::-1] #top 9 
            lr = np.squeeze(lr[sorted_by_clearance], axis=1)
            
            if self.transfrom == True:
                #brightness and gamma shift
                gamma_scale = np.random.uniform(low=0.8, high=1.2)
                brightness_scale = np.random.uniform(low=0.8, high=1.2)
                lr,hr = do_gamma(lr,hr , gamma_scale)
                lr,hr = do_brightness_multiply(lr,hr, brightness_scale)
                #random crop and rotate
                hr = Image.fromarray(hr)
                lr = np_3d_to_PIL(lr)   #convert lr to PIL in order to perform rotation/crop     
                lr , hr = random_crop(lr, hr)

                lr, hr = flip_lr_hr(lr, hr)
                lr , hr = rotate_lr_hr(lr , hr)

                lr = PIL_3d_to_np(lr)
                hr = np.array(hr)
            
            hr = np.expand_dims(hr, 0)
            lr, hr = torch.tensor(lr), torch.tensor(hr)
            #upsamling
            lr = F.interpolate(lr.unsqueeze(1), size = [384,384], mode =  'bicubic', align_corners = True).squeeze(1)
            hr = F.interpolate(hr.unsqueeze(0), size = [384,384], mode =  'bicubic', align_corners = True).squeeze(0) 
            return lr, sm[sorted_by_clearance], hr
        
        elif self.mode == 'test':
            lr, sm = get_lr_sm_from_scene(self.data_path[index])
            if self.input_transform:
                input_image = self.input_transform(input_image)  
            sm = np.squeeze(sm, axis=1)
            #select n lr images based on degree of clearance 
            sorted_by_clearance = np.sum(sm, axis = (1,2)).argsort()[-9:][::-1] #top 9 
            lr = np.squeeze(lr, axis=1)
            
            return lr[sorted_by_clearance], sm[sorted_by_clearance]
        
        else:  raise Exception('mode can only be train or test')


    def __len__(self):
        return len(self.data_path)
    
    
