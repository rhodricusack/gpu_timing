import numpy as np
import math
from scipy.ndimage import gaussian_filter 
import numbers
from PIL import Image
from torchvision import transforms
import torch
from torch import Tensor


class TinyEyes(torch.nn.Module):

    """Apply TinyEyes Transformation to the given PIL Image post CenterCrop.
    Args:
        age (str): 
        width (float or int): 
        dist (float or int): 
    """
    
    # Define static attributes
    
    _convert_rgb2lms=np.array([[0.05059983, 0.08585369, 0.00952420],
                               [0.01893033, 0.08925308, 0.01370054],
                               [0.00292202, 0.00975732, 0.07145979]])  
    
    # convert array to tensor
    _convert_rgb2lms_t = torch.from_numpy(_convert_rgb2lms).float()
    
    _convert_lms2opp=np.array([[  0.5000,     0.5,       0],
                               [ -0.6690,  0.7420, -0.0270],
                               [ -0.2120, -0.3540,  0.9110]])
    
    # convert array to tensor
    _convert_lms2opp_t = torch.from_numpy(_convert_lms2opp).float()
    
     # To move the other way (from LMS to RGB) we invert that transform
    _convert_lms2rgb=np.linalg.pinv(_convert_rgb2lms)

    # convert _convert_lms2rgb to tensor
    _convert_lms2rgb_t = torch.from_numpy(_convert_lms2rgb).float()
    
    _convert_opp2lms=np.linalg.inv(_convert_lms2opp)

    # convert _convert_opp2lms to tensor
    _convert_opp2lms_t = torch.from_numpy(_convert_opp2lms).float()
    
    # Get model of BLUR
    # from dict of ages and model values
    # Values calculated by Prof Alex Wade from developmental literature
    _modelByAge = {'week0':   np.array([0.6821,100, 1000]),
                    'week4':  np.array([0.48,4.77,100]),
                    'week8':  np.array([0.24, 2.4, 4]),
                    'week12': np.array([0.1,0.53,2]),
                    'week24': np.array([0.04,0.12,1]),
                    'adult':  np.array([0.01,0.015,0.02])}
    
    _possible_ages = ['week0', 'week4', 'week8', 'week12', 'week24', 'adult']
        

    def __init__(self, age, width, dist, imp='cpu'):
        
        """Get parameters for TinyEyes Transformation.

        Args:
            age (str): 
            width (float or int): 
            dist (float or int): 

        """
        
        super().__init__() 
            
        if age not in TinyEyes._possible_ages:
            raise ValueError(f"Please choose one of the following possible ages: {TinyEyes._possible_ages}")
        self.age = age
        
        if isinstance(width, numbers.Number):
            if width < 0:
                raise ValueError("width must be a positive value (cm)")
            self.width = width
        else:
            raise TypeError("width must be a positive number (cm)")  

        if isinstance(dist, numbers.Number):
            if dist < 0:
                raise ValueError("dist must be a positive value (cm)")
            self.dist = dist
        else:
            raise TypeError("dist must be a positive number (cm)")
        
        self.modelForAge = TinyEyes._modelByAge[age]

        # Add variable for implementation - 'gpu' or 'cpu'
        # default to cpu
        if imp in ['cpu', 'gpu']:
            self.imp = imp
        else:
            raise ValueError("Please choose one of the following possible implementations: 'cpu' or 'gpu'")

        self.n_images_processed = 0


    def calc_ppd_from_params(self, px_size):
        ratio =(self.width/2)/self.dist
        t = 2 * math.degrees((math.atan(ratio)))
        ppd = px_size/t
        return ppd
    
    
    def calc_blur_from_params(self, img):
        # TO DO:Update this text
        """Calculate parameters for TinyEyes Transformation

        Args:
            img (PIL Image or torch tensor): Image to be Transformed.
            
        Returns:
            numpy array: model of Blur for given age, viewing distance and image size
        """
        
        if isinstance(img, Image.Image):      
            img_w, img_h = img.size
           
        elif isinstance(img, Tensor):
            img_c, img_w, img_h = tuple(img.size())
        
        if img_w != img_h:
            raise ValueError("Image must be a square, try using CenterCrop transformation with single input param eg CentreCrop(224) first. This ensures all images are the same size!")
        
        px_size = img_w  # Which also equals height

        # Calculate channelwise blur based on input parameters
        ppd = self.calc_ppd_from_params(px_size)
        # print(f'Pixels per degree: {ppd}, approx. {ppd/2} cycles per degree')
        lrb = self.modelForAge * ppd
        channelwise_blur = np.around(lrb, 2)
        # print(f'Blur values: {channelwise_blur}')

        return channelwise_blur



    def forward(self, img):
        """
        Args:
            img (PIL Image or torch tensor): Image to be 

        Returns:
            PIL Image: Image with TinyEyes Transformation applied
        """

        if self.imp == 'gpu':
            if isinstance(img,Image.Image):
               img = transforms.ToTensor()(img)

            if isinstance(img, Tensor):
                if img.ndim == 3:
                    img = img[np.newaxis,:,:,:]

                # Create an empty tinyeyes_img to store the output
                tinyeyes_img = torch.zeros_like(img)

                # Take a batch of images as an input
                channelwise_blur = self.calc_blur_from_params(img[0,:,:,:])
                
                tinyeyes_img = self.blur_torch_image(img, channelwise_blur)


        elif self.imp == 'cpu':
            # # else:
            channelwise_blur = self.calc_blur_from_params(img)
        
            if isinstance(img, Image.Image):
                tinyeyes_img = self.blur_np_image(img, channelwise_blur)

            if isinstance(img, Tensor):
                tinyeyes_img = self.blur_torch_image(img, channelwise_blur)

        return tinyeyes_img



    def blur_np_image(self, img, channelwise_blur):

        # Convert PIL img to np array
        imgNP = np.array(img).astype('uint8')
        
        x,y,z = imgNP.shape

        # Converted to row format ,new shape should be 3 x (x*y)
        imgReshaped = imgNP.transpose(2,0,1).reshape(3,-1).astype(np.float)

        # Convert to LMS space... and then OPPonent space
        # Before we do that we convert RGB to RGB >contrast< around the mid-point (128, a grey screen)
        # LMS or cone opponent spaces work in contrast units. 
        imgLMS = np.matmul(self._convert_rgb2lms, (imgReshaped-128)/128)
        imgOPP = np.matmul(self._convert_lms2opp,  imgLMS)

        imgBLUR = imgOPP.reshape(z,x,y)

        # Apply appropriate blur to each cahnnel in opponent space
        for i in range(3):
            imgBLUR[i,:] = gaussian_filter(imgBLUR[i,:], channelwise_blur[i]) 

        imgBLUR = imgBLUR.reshape(z,x*y) # back to a 2d array

        imgOUTLMS = np.matmul(self._convert_opp2lms, imgBLUR) #Convert back to LMS

        # LMS or cone opponent spaces work in contrast units. 
        # But RGB is a number between 0 and 255. We need to convert back...
        imgOUT=np.matmul(self._convert_lms2rgb,imgOUTLMS)*128+128 

        imgOUT = imgOUT.reshape(z,x,y).transpose(1,2,0).astype(int)

        # Threshold between [0, 255] to prevent out of range artefacts
        # Question: Should I be thresholding oR re-scaling the values?
        imgOUT[imgOUT<0]=0
        imgOUT[imgOUT>255]=255

        # Convert back to PIL image
        tinyeyes_img = Image.fromarray(imgOUT.astype('uint8'), 'RGB')
        
        return tinyeyes_img
    


    def blur_torch_image(self, img, channelwise_blur):
    
        if self.imp == 'gpu' and img.ndim==4:
            # CUDA on batch
            n,c,x,y = img.size()
            # A function of image size, we will need this for our gaussian filter later
            max_allowable_kernel_size = (x * 2) -1

            imgBLUR = torch.empty((c,n*x*y), device='cuda') # note colour first here!

            imgReshaped = img.permute([1,0,2,3]).reshape(3,-1).float()
            imgLMS = torch.matmul(self._convert_rgb2lms_t.to('cuda'), (imgReshaped-0.5)/0.5) 
            imgOPP = torch.matmul(self._convert_lms2opp_t.to('cuda'),  imgLMS.to('cuda'))

            for channel in range(c):
                # Dynamically update kernel size based on sigma (blur value) as defined in PIL implementation above (scipy.ndimage.gaussian_filter)
                kernel_size = (2*round(4*channelwise_blur[channel]))+1
                if kernel_size > max_allowable_kernel_size:
                    kernel_size = max_allowable_kernel_size

                imgBLUR[channel,:]  = transforms.GaussianBlur(kernel_size=kernel_size,
                                        sigma=(channelwise_blur[channel]
                                            ))(imgOPP[channel,:].reshape(n,x,y)).view(-1)
                                        
                # Convert back to RGB
                imgOUTLMS = torch.matmul(self._convert_opp2lms_t.to('cuda'), imgBLUR ) #Convert back to LMS
                imgOUT=torch.matmul(self._convert_lms2rgb_t.to('cuda') ,imgOUTLMS)*0.5+0.5 

                # Question: does clipping the values make sense if we are going to rescale through normalisation anyway?
                imgOUT[imgOUT<0]=0
                imgOUT[imgOUT>1]=1

                imgOUT = imgOUT.reshape(c,n,x,y).permute([1,0,2,3])
        else:
            raise('Not supported')
        
        # Output tensor
        return imgOUT


    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'age={0}'.format(self.age)
        format_string += ', width={0}'.format(self.width)
        format_string += ', dist={0})'.format(self.dist)
        return format_string



