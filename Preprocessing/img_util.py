import PIL.Image
import numpy as np
import io
import torch

def get_tensor(image, dim = 3, dtype = 'float', device = None):
    image = get_np(image)

    if dim == 2:
        image = image[:, :, 0] / 255.
    elif dim == 3:
        image = image[:, :, : 1] / 255.

    if dtype == 'float':
        image = torch.tensor(image, dtype = torch.float32, device = device)
    elif dtype == 'long':
        image = torch.tensor(image, dtype = torch.long, device = device)

    return image
        

def get_np(image):
    if isinstance(image, bytes):
        image = get_PIL(image)
        image = np.array(image)
        
    elif isinstance(image, np.ndarray):
        # Case float
        if image.dtype != 'uint8':
            # Case between 0. and 1.
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            # Case between 0. and 255.
            else:
                image = image.astype(np.uint8)
        
        # Case 2 dimensionnal
        if len(image.shape) == 2:
            image = np.repeat(image[:, :, np.newaxis], 3, axis = 2)
            
        # Case 3 dimensionnal
        if len(image.shape) == 3 and image.shape[2] == 1:
            image = np.repeat(image, 3, axis = 2)
    
    elif isinstance(image, torch.Tensor):
        image = image.cpu().detach().numpy()
        # Case float
        if image.dtype != 'uint8':
            # Case between 0. and 1.
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            # Case between 0. and 255.
            else:
                image = image.astype(np.uint8)
        
        # Case 2 dimensionnal
        if len(image.shape) == 2:
            image = np.repeat(image[:, :, np.newaxis], 3, axis = 2)
            
        # Case 3 dimensionnal
        if len(image.shape) == 3 and image.shape[2] == 1:
            image = np.repeat(image, 3, axis = 2)
    
    elif isinstance(image, PIL.Image.Image):
        image = np.array(image)
    
    return image

def get_PIL(image):
    if isinstance(image, bytes):
        with io.BytesIO(image) as stream:
            image = PIL.Image.open(stream)
            image.load()
            
    elif isinstance(image, np.ndarray):
        image = PIL.Image.fromarray(image)
            
    elif isinstance(image, torch.Tensor):
        image = get_np(image)
        image = get_PIL(image)
        
    elif isinstance(image, PIL.Image.Image):
        pass
    return image

def get_bytes(image, format = 'png'):
    if isinstance(image, PIL.Image.Image):
        image = get_PIL(image)

        with io.BytesIO() as stream:
            image.save(stream, format = format)
            image = stream.getvalue()
        
    elif isinstance(image, bytes):
        pass

    return image