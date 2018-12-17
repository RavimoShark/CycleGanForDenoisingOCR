import torch
import torchvision
import random
import glob
import PIL.Image
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 

# Import stuff
import numpy as np
import pandas as pd
import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

all_files = glob.glob('**', recursive=True)
google_fonts = [e for e in all_files if '.ttf' in e and 'google_fonts' in e]
regular_fonts = sorted([e for e in google_fonts if '-Regular.' in e])
italic_fonts = sorted([e for e in google_fonts if '-Italic.' in e])

defuncion_fonts = ['google_fonts/fonts-master/apache/opensanshebrew/OpenSansHebrew-Italic.ttf',
 'google_fonts/fonts-master/apache/tinos/Tinos-Italic.ttf',
 'google_fonts/fonts-master/ofl/abeezee/ABeeZee-Italic.ttf',
 'google_fonts/fonts-master/ofl/archivovfbeta/ArchivoVFBeta-Italic.ttf',
 'google_fonts/fonts-master/ofl/asapcondensed/AsapCondensed-Italic.ttf',
 'google_fonts/fonts-master/ofl/asapvfbeta/AsapVFBeta-Italic.ttf',
 'google_fonts/fonts-master/ofl/chivo/Chivo-Italic.ttf',
 'google_fonts/fonts-master/ofl/creteround/CreteRound-Italic.ttf',
 'google_fonts/fonts-master/ofl/exo2/Exo2-Italic.ttf',
 'google_fonts/fonts-master/ofl/gudea/Gudea-Italic.ttf',
 'google_fonts/fonts-master/ofl/istokweb/IstokWeb-Italic.ttf',
 'google_fonts/fonts-master/ofl/k2d/K2D-Italic.ttf',
 'google_fonts/fonts-master/ofl/karla/Karla-Italic.ttf',
 'google_fonts/fonts-master/ofl/lato/Lato-Italic.ttf',
 'google_fonts/fonts-master/ofl/marvel/Marvel-Italic.ttf',
 'google_fonts/fonts-master/ofl/merriweather/Merriweather-Italic.ttf',
 'google_fonts/fonts-master/ofl/merriweathersans/MerriweatherSans-Italic.ttf',
 'google_fonts/fonts-master/ofl/prompt/Prompt-Italic.ttf',
 'google_fonts/fonts-master/ofl/prozalibre/ProzaLibre-Italic.ttf',
 'google_fonts/fonts-master/ofl/ptsans/PT_Sans-Web-Italic.ttf',
 'google_fonts/fonts-master/ofl/raleway/Raleway-Italic.ttf',
 'google_fonts/fonts-master/ofl/ropasans/RopaSans-Italic.ttf',
 'google_fonts/fonts-master/ofl/sarabun/Sarabun-Italic.ttf',
 'google_fonts/fonts-master/ofl/scada/Scada-Italic.ttf',
 'google_fonts/fonts-master/ofl/unna/Unna-Italic.ttf']

# Modeling
# --------

def get_weights(image_array):
    weights_white = np.clip(10 - cv2.distanceTransform(image_array[:, :, 0], cv2.DIST_L2, 3), 1, 10)
    weights_black = np.clip(10 - cv2.distanceTransform(~image_array[:, :, 0], cv2.DIST_L2, 3), 1, 10)
    weights = weights_white
    weights[image_array[:, :, 0] == 0] = weights_black[image_array[:, :, 0] == 0]
    return weights


# Generation
# ----------

def sequence_to_map(sequence):
    map = {}
    inverse_map = {}
    for e in sequence:
        map.setdefault(e, len(map) + 1)
        inverse_map.setdefault(map[e], e)
        
    inverse_map[0] = '!'
    map['!'] = 0
    return map, inverse_map

def generate_image(base_dimension, font_path, text):
    image = PIL.Image.new("RGB", base_dimension, color = (255, 255, 255))
    font = ImageFont.truetype(font_path, base_dimension[1])
    random_factor = random.random() * 0.4 + 0.6
    target_pixel_size = base_dimension[1] * 0.8 * random_factor
    real_pixel_size = font.getsize(text)[1]
    new_size = round(base_dimension[1] * (target_pixel_size / real_pixel_size))
    font = ImageFont.truetype(font_path, new_size)
    real_pixel_size = font.getsize(text)
    margin = (base_dimension[0] - real_pixel_size[0], base_dimension[1] - real_pixel_size[1])
    random_positon = (random.randint(0, margin[0]), random.randint(0, margin[1]))
    
    draw = ImageDraw.Draw(image)
    draw.text(random_positon, text, 0, font = font)
    return np.array(image)

def generate_color_image(base_dimension, font_path, text, color_map):
    image = PIL.Image.new("RGB", base_dimension, color = (255, 255, 255))
    font = ImageFont.truetype(font_path, base_dimension[1])
    random_factor = random.random() * 0.4 + 0.6
    target_pixel_size = base_dimension[1] * 0.8 * random_factor
    real_pixel_size = font.getsize(text)[1]
    new_size = round(base_dimension[1] * (target_pixel_size / real_pixel_size))
    font = ImageFont.truetype(font_path, new_size)
    real_pixel_size = font.getsize(text)
    margin = (base_dimension[0] - real_pixel_size[0], base_dimension[1] - real_pixel_size[1])
    random_positon = (random.randint(0, margin[0]), random.randint(0, margin[1]))

    draw = ImageDraw.Draw(image)
    for i in range(len(text), 0, -1):    
        draw.text(random_positon, text[:i], fill = color_map[text[i - 1]], font = font)

    return np.array(image)

def generate_data_point(text = None, font = None):
    if text is None:
        text = generate_random_text()
    
    if font is None:
        font = random.choice(true_type_fonts)
    
    target = generate_image((2048, 64), font, text)
    target = small_perturbation(target)
    input = big_perturbation(target)
    
    weights = get_weights(target)

    return input, target, weights, font, text

def generate_color_data_point(text, font, color_map):
    target = generate_image((2048, 64), font, text, color_map)
    target = small_perturbation_color(target, color_map)
    target[target[:, :, 0] == 255] = 0
    input = target.copy()
    input = 
    input = big_perturbation(target)
    
    weights = get_weights(target)

    return input, target, weights, font, text

# Perturbations
# -------------

def big_perturbation(image_array):
    image_array = image_array.copy()
    
    image_array = resize_perturbation(image_array)
    
    random_transfo = random.randint(0, 3)
    if random_transfo == 0:
        image_array = erode_perturbation(image_array)
    elif random_transfo == 1:
        image_array = dilate_perturbation(image_array)
    
    if random.random() < 0.5:
        permutation_proportion = random.random() * 0.10
        image_array = permute_perturbation(image_array, proportion = permutation_proportion)

    return image_array

def small_perturbation(image_array):
    image_array = binarize(image_array)
    image_array = elastic_transform(image_array, image_array.shape[1] * 0.6, image_array.shape[1] * 0.05, image_array.shape[1] * 0.0)
    image_array = binarize(image_array)
    return image_array

def small_perturbation_color(image_array, color_map):
    image_array = binarize_color(image_array, color_map)
    image_array = elastic_transform(image_array, image_array.shape[1] * 0.6, image_array.shape[1] * 0.05, image_array.shape[1] * 0.0)
    image_array = binarize_color(image_array, color_map)
    return image_array

def resize_perturbation(image_array, min_factor = 0.3):
    factor = random.random() * (1 - min_factor) + min_factor
    new_size = (int(image_array.shape[1] * factor), int(image_array.shape[0] * factor))
    original_size = image_array.shape[:2][::-1]
    image = get_PIL(image_array).resize(new_size, resample=PIL.Image.BILINEAR)
    image = get_PIL(binarize(get_np(image)))
    image_array = get_np(image.resize(original_size, resample=PIL.Image.BILINEAR))
    image_array = binarize(image_array)
    return image_array


def permute_perturbation(image_array, proportion = 0.2):
    permutation = ((np.random.random(image_array.shape[:2]) <= proportion) * 255).astype(np.uint8)
    permutation = dilate_perturbation(permutation, 4)[:, :, 0]
    
    permutation_2 = ((np.random.random(image_array.shape[:2]) <= 0.7) * 255).astype(np.uint8)
    permutation[permutation_2 == 255] = 0
    
    image_array[permutation == 255, :] = 255 - image_array[permutation == 255, :]
    
    return image_array

def erode_perturbation(image_array, range_max = 5):
    image_array = get_np(image_array)
    erode_value = random.randint(1, range_max)
    if erode_value > 1:
        image_array = erode(image_array, erode_value)
    return image_array

def dilate_perturbation(image_array, range_max = 3):
    image_array = get_np(image_array)
    erode_value = random.randint(1, range_max)
    if erode_value > 1:
        image_array = dilate(image_array, erode_value)
    return image_array

# Transformations
# ---------------
    
def binarize(image_array):
    image_array[image_array > 127] = 255
    image_array[image_array <= 127] = 0
    return image_array

def binarize_color(image_array, color_map):
    mask = np.zeros(image_array.shape[:2], dtype = np.uint8)
    for color in color_map.values():
        mask[image_array == mask] = 1
    image_array[mask == 0] = 255
    return image_array

def show(image_array):
    return PIL.Image.fromarray(image_array)

# Function to distort image
def elastic_transform(image_array, alpha, sigma, alpha_affine, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    image = image_array
        
    shape = image.shape
    shape_size = shape[:2]
    
    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

    image_array = map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
    return image_array

def erode(image_array, size):
    image_array = get_np(image_array)
    return cv2.erode(image_array, np.ones((size, size)))

def dilate(image_array, size):
    image_array = get_np(image_array)
    return cv2.dilate(image_array, np.ones((size, size)))

def median(image_array, size):
    image_array = get_np(image_array)
    return cv2.medianBlur(image_array, size)

def diff(first_image, second_image):
    result_image = np.empty(first_image.shape, dtype = np.uint8)
    result_image[first_image == second_image] = 255
    result_image[first_image != second_image] = 0
    return result_image

def sample(image_array, proportion = 0.7):
    points = (255 - image_array[:, :, 0]).nonzero()
    if points[0].shape[0] > 0:
        selected_indices = np.random.choice(np.arange(points[0].shape[0]), round(points[0].shape[0] * (1 - proportion)))
        image_array[points[0][selected_indices], points[1][selected_indices], :] = 255
    return image_array

def apply(first_image, second_image):
    image_array = first_image
    image_array[image_array[:, :, 0] != second_image[:, :, 0], :] = 0
    return image_array

def sample_in_diff_and_add(base_image, different_image):
    sampled_image = sample(different_image)
    return apply(base_image, sampled_image)

