from model_util import *
from img_util import *

def NUMBERS_generate_random_text():
    charsets = [#"abcdefghijklmnopqrstuvwxyz     -/.'",
                "0123456789-/",
                #"abcdefghijklmnopqrstuvwxyz     -/.'",
                #"ABCDEFGHIJKLMNOPQRSTUVWXYZ     -/.'",
                #"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ          -/.'",
                #"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ          -/.'"
    ]
    charset = random.choice(charsets)
    length = random.randint(10, 20)
    string = "".join([random.choice(charset) for _ in range(length)])
    return string

def NUMBERS_generate_maps():
    words = '0123456789-/'
    token_to_index, index_to_token = sequence_to_map(list(words))
    return token_to_index, index_to_token

def DATE_generate_random_text():
    words = 'uno dos tres cuatro cinco seis siete ocho nueve dies once doce trece catorce quince dieciseis diecisiete dieciocho diecinueve vinte veinte trenta cuarenta cincuenta sesenta setenta ochenta noventa ciente mil'.split()
    length = random.randint(5, 9)
    string = " ".join([random.choice(words) for _ in range(length)])
    return string

def DATE_generate_maps():
    words = 'uno dos tres cuatro cinco seis siete ocho nueve dies once doce trece catorce quince dieciseis diecisiete dieciocho diecinueve vinte veinte trenta cuarenta cincuenta sesenta setenta ochenta noventa ciente mil'.split()
    token_to_index, index_to_token = sequence_to_map(list(" ".join(words)))
    return token_to_index, index_to_token


def ALL_generate_random_text():
    charsets = ["abcdefghijklmnopqrstuvwxyz     -/.'",
                "0123456789-/",
                "abcdefghijklmnopqrstuvwxyz     -/.'",
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ     -/.'",
                "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ          -/.'",
                "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ          -/.'"
    ]
    charset = random.choice(charsets)
    length = random.randint(10, 20)
    string = "".join([random.choice(charset) for _ in range(length)])
    return string

def ALL_generate_maps():
    charsets = ["abcdefghijklmnopqrstuvwxyz     -/.'",
                "0123456789-/",
                "abcdefghijklmnopqrstuvwxyz     -/.'",
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ     -/.'",
                "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ          -/.'",
                "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ          -/.'"
    ]
    token_to_index, index_to_token = sequence_to_map(list("".join(charsets)))
    return token_to_index, index_to_token


# Torch datasets
# --------------

class ImageDataset(torch.utils.data.Dataset):
    """
    Generated dataset for a mono masking process.
    The dataset distribution is customizable 
    """
    def __init__(self, size, text_function, map_function, fonts, device = None):
        self.text_function = text_function
        self.fonts = fonts
        self.size = size
        self.device = device
        token_to_index, index_to_token = map_function()
        self.token_to_index = token_to_index
        self.index_to_token = index_to_token

    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        sample = {}
        while True:
            try:
                input, target, weights, font, text = generate_data_point(text = self.text_function(), font = random.choice(self.fonts))
                break
            except KeyboardInterrupt:
                break
            except:
                print("Error")

        input, target = input[:, :, :1], target[:, :, 0]
        sample['input'] = torch.tensor(input.astype(np.float32) / 255, device = self.device)
        sample['target'] = torch.tensor(target / 255, dtype = torch.long, device = self.device)
        #sample['target_text'] = torch.tensor([self.token_to_index[c] for c in text], dtype = torch.long, device = self.device)
        sample['weights'] = torch.tensor(weights.astype(np.float32), device = self.device)
        sample['input_length'] = torch.tensor(2048, dtype = torch.long, device = self.device)
        sample['target_length'] = torch.tensor(len(text), dtype = torch.long, device = self.device)
        sample['font'] = font
        sample['text'] = text
        
        return sample
    
    
class ImageDatasetColor(torch.utils.data.Dataset):
    """
    Generated dataset for a mono masking process.
    The dataset distribution is customizable 
    """
    def __init__(self, size, text_function, map_function, fonts, device = None):
        self.text_function = text_function
        self.fonts = fonts
        self.size = size
        self.device = device
        token_to_index, index_to_token = map_function()
        self.token_to_index = token_to_index
        self.index_to_token = index_to_token
        
        self.color_map = {k : tuple([i + 1] + list(np.random.randint(0, 255, (2,)))) for i, k in enumerate(sorted(token_to_index.keys()))}

    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        sample = {}
        while True:
            try:
                input, target, weights, font, text = generate_color_data_point(text = self.text_function(), font = random.choice(self.fonts), color_map = self.color_map)
                break
            except KeyboardInterrupt:
                break
            except:
                print("Error")

        input, target = input[:, :, :1], target[:, :, 0]
        sample['input'] = torch.tensor(input.astype(np.float32) / 255, device = self.device)
        sample['target'] = torch.tensor(target / 255, dtype = torch.long, device = self.device)
        sample['target_text'] = torch.tensor([self.token_to_index[c] for c in text], dtype = torch.long, device = self.device)
        sample['weights'] = torch.tensor(weights.astype(np.float32), device = self.device)
        sample['input_length'] = torch.tensor(2048, dtype = torch.long, device = self.device)
        sample['target_length'] = torch.tensor(len(text), dtype = torch.long, device = self.device)
        sample['font'] = font
        sample['text'] = text
        
        return sample