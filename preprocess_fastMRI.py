import os
import pydicom
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import pickle
import json

import nibabel as nib
import os
from matplotlib import pyplot as plt
import numpy as np
import torch
from typing import Tuple
from fftc import ifft2c_new as ifft2
import pickle

dcm_path = '/disk/yanyunlu/data/fastMRI_brain_DICOM'
save_path = '/disk/yanyunlu/data/fastMRI_brain_pkl_threemodals'

def complex_center_crop(data: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    """
    Apply a center crop to the input image or batch of complex images.

    Args:
        data: The complex input tensor to be center cropped. It should have at
            least 3 dimensions and the cropping is applied along dimensions -3
            and -2 and the last dimensions should have a size of 2.
        shape: The output shape. The shape should be smaller than the
            corresponding dimensions of data.

    Returns:
        The center cropped image
    """
    if not (0 < shape[0] <= data.shape[-3] and 0 < shape[1] <= data.shape[-2]):
        raise ValueError("Invalid shapes.")

    w_from = (data.shape[-3] - shape[0]) // 2
    h_from = (data.shape[-2] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]

    return data[..., w_from:w_to, h_from:h_to, :]


def save_pkl(pkldata, name):
    os.makedirs(pkl_path, exist_ok=True)
    pkl_file = os.path.join(pkl_path, name+'.pkl')
    with open(pkl_file, 'wb') as f:
        pickle.dump(pkldata, f)

def fft2(img):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img)))

def to_tensor(data: np.ndarray) -> torch.Tensor:
    """
    Convert numpy array to PyTorch tensor.

    For complex arrays, the real and imaginary parts are stacked along the last
    dimension.

    Args:
        data: Input numpy array.

    Returns:
        PyTorch version of data.
    """
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)

    return torch.from_numpy(data)

def complex_abs(data: torch.Tensor) -> torch.Tensor:
    """
    Compute the absolute value of a complex valued input tensor.

    Args:
        data: A complex valued tensor, where the size of the final dimension
            should be 2.

    Returns:
        Absolute value of data.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    return (data ** 2).sum(dim=-1).sqrt()

def process(img):
    img = img.transpose(1, 0)
    kspace = fft2(img)
    kspace = to_tensor(kspace)
    kspace = complex_center_crop(kspace, (200, 200))
    image = ifft2(kspace)
    image = complex_abs(image)
    return image

def show(img, file):
    os.makedirs('fig', exist_ok=True)
    plt.figure()
    plt.imshow(img, 'gray')
    plt.savefig(os.path.join('fig', file))

def load_scan(path):
    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))

    slice_dict = {}
    for s in slices:

        data = np.array(s.pixel_array, dtype='float32')
        #data = process(data)
        # data = data.numpy()

        if s.SeriesDescription in slice_dict.keys():
            slice_dict[s.SeriesDescription].append(data)
        else:
            slice_dict[s.SeriesDescription] = []
            slice_dict[s.SeriesDescription].append(data)
    return slice_dict

def main():
    data_list = []
    json_file = 'data.json'

    with open(json_file, 'r') as  f:
        datas = json.load(f)
        for d in tqdm(datas):
            k = list(d.keys())
            modalitys = d[k[0]]
            names = modalitys.keys()
            if 'AX T1' in names and 'AX T2' in names and 'AX T1 POST' in names:
                if modalitys['AX T1'] == modalitys['AX T2'] and modalitys['AX T1'] == modalitys['AX T1 POST'] and modalitys['AX T1'] ==[16, 320, 320]:
                    data_list.append(k[0])

    print(len(data_list))

    for dir in tqdm(data_list):
        dcm_file_path = os.path.join(dcm_path, dir)
        slice_dict = load_scan(dcm_file_path)
        os.makedirs(save_path, exist_ok=True)
        pkl_data = {}
        for key in slice_dict.keys():
            try:
                data = np.stack([slice_dict[key][i] for i in range(len(slice_dict[key]))])

                if key=='AX T1':
                    pkl_data['t1'] = data
                if key == 'AX T2':
                    pkl_data['t2'] = data
                if key == 'AX T1 POST':
                    pkl_data['post'] = data
            except:
                continue


        file_path = os.path.join(save_path, dir + '.pkl')
        with open(file_path, 'wb') as f:
            pickle.dump(pkl_data, f)




if __name__ == '__main__':
    main()