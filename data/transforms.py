"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from typing import Dict, Optional, Sequence, Tuple, Union

from data import fastmri
import numpy as np
import torch
import random
from .subsample import MaskFunc
from typing import Tuple, Union, Callable

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


def tensor_to_complex_np(data: torch.Tensor) -> np.ndarray:
    """
    Converts a complex torch tensor to numpy array.

    Args:
        data: Input data to be converted to numpy.

    Returns:
        Complex numpy version of data.
    """
    data = data.numpy()

    return data[..., 0] + 1j * data[..., 1]


def apply_mask(
    data: torch.Tensor,
    mask_func: MaskFunc,
    seed: Optional[Union[int, Tuple[int, ...]]] = None,
    padding: Optional[Sequence[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Subsample given k-space by multiplying with a mask.

    Args:
        data: The input k-space data. This should have at least 3 dimensions,
            where dimensions -3 and -2 are the spatial dimensions, and the
            final dimension has size 2 (for complex values).
        mask_func: A function that takes a shape (tuple of ints) and a random
            number seed and returns a mask.
        seed: Seed for the random number generator.
        padding: Padding value to apply for mask.

    Returns:
        tuple containing:
            masked data: Subsampled k-space data
            mask: The generated mask
    """
    shape = np.array(data.shape)
    shape[:-3] = 1
    mask = mask_func(shape, seed)
    if padding is not None:
        mask[:, :, : padding[0]] = 0
        mask[:, :, padding[1] :] = 0  # padding value inclusive on right of zeros

    masked_data = data * mask + 0.0  # the + 0.0 removes the sign of the zeros

    return masked_data, mask


def mask_center(x: torch.Tensor, mask_from: int, mask_to: int) -> torch.Tensor:
    """
    Initializes a mask with the center filled in.

    Args:
        mask_from: Part of center to start filling.
        mask_to: Part of center to end filling.

    Returns:
        A mask with the center filled.
    """
    mask = torch.zeros_like(x)
    mask[:, :, :, mask_from:mask_to] = x[:, :, :, mask_from:mask_to]

    return mask


def center_crop(data: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    """
    Apply a center crop to the input real image or batch of real images.

    Args:
        data: The input tensor to be center cropped. It should
            have at least 2 dimensions and the cropping is applied along the
            last two dimensions.
        shape: The output shape. The shape should be smaller
            than the corresponding dimensions of data.

    Returns:
        The center cropped image.
    """
    if not (0 < shape[0] <= data.shape[-2] and 0 < shape[1] <= data.shape[-1]):
        raise ValueError("Invalid shapes.")

    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]

    return data[..., w_from:w_to, h_from:h_to]


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


def center_crop_to_smallest(
    x: torch.Tensor, y: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply a center crop on the larger image to the size of the smaller.

    The minimum is taken over dim=-1 and dim=-2. If x is smaller than y at
    dim=-1 and y is smaller than x at dim=-2, then the returned dimension will
    be a mixture of the two.

    Args:
        x: The first image.
        y: The second image.

    Returns:
        tuple of tensors x and y, each cropped to the minimim size.
    """
    smallest_width = min(x.shape[-1], y.shape[-1])
    smallest_height = min(x.shape[-2], y.shape[-2])
    x = center_crop(x, (smallest_height, smallest_width))
    y = center_crop(y, (smallest_height, smallest_width))

    return x, y


def normalize(
    data: torch.Tensor,
    mean: Union[float, torch.Tensor],
    stddev: Union[float, torch.Tensor],
    eps: Union[float, torch.Tensor] = 0.0,
) -> torch.Tensor:
    """
    Normalize the given tensor.

    Applies the formula (data - mean) / (stddev + eps).

    Args:
        data: Input data to be normalized.
        mean: Mean value.
        stddev: Standard deviation.
        eps: Added to stddev to prevent dividing by zero.

    Returns:
        Normalized tensor.
    """
    return (data - mean) / (stddev + eps)


def normalize_instance(
    data: torch.Tensor, eps: Union[float, torch.Tensor] = 0.0
) -> Tuple[torch.Tensor, Union[torch.Tensor], Union[torch.Tensor]]:
    """
    Normalize the given tensor  with instance norm/

    Applies the formula (data - mean) / (stddev + eps), where mean and stddev
    are computed from the data itself.

    Args:
        data: Input data to be normalized
        eps: Added to stddev to prevent dividing by zero.

    Returns:
        torch.Tensor: Normalized tensor
    """
    mean = data.mean()
    std = data.std()

    return normalize(data, mean, std, eps), mean, std

def augment_gamma(data_sample, gamma_range=(0.5, 2), invert_image=False, epsilon=1e-7, per_channel=False,
                  retain_stats: Union[bool, Callable[[], bool]] = False):
    if invert_image:
        data_sample = - data_sample

    if not per_channel:
        retain_stats_here = retain_stats() if callable(retain_stats) else retain_stats
        if retain_stats_here:
            mn = data_sample.mean()
            sd = data_sample.std()
        if np.random.random() < 0.5 and gamma_range[0] < 1:
            gamma = np.random.uniform(gamma_range[0], 1)
        else:
            gamma = np.random.uniform(max(gamma_range[0], 1), gamma_range[1])
        minm = data_sample.min()
        rnge = data_sample.max() - minm
        data_sample = np.power(((data_sample - minm) / float(rnge + epsilon)), gamma) * rnge + minm
        if retain_stats_here:
            data_sample = data_sample - data_sample.mean()
            data_sample = data_sample / (data_sample.std() + 1e-8) * sd
            data_sample = data_sample + mn
    else:
        for c in range(data_sample.shape[0]):
            retain_stats_here = retain_stats() if callable(retain_stats) else retain_stats
            if retain_stats_here:
                mn = data_sample[c].mean()
                sd = data_sample[c].std()
            if np.random.random() < 0.5 and gamma_range[0] < 1:
                gamma = np.random.uniform(gamma_range[0], 1)
            else:
                gamma = np.random.uniform(max(gamma_range[0], 1), gamma_range[1])
            minm = data_sample[c].min()
            rnge = data_sample[c].max() - minm
            data_sample[c] = np.power(((data_sample[c] - minm) / float(rnge + epsilon)), gamma) * float(rnge + epsilon) + minm
            if retain_stats_here:
                data_sample[c] = data_sample[c] - data_sample[c].mean()
                data_sample[c] = data_sample[c] / (data_sample[c].std() + 1e-8) * sd
                data_sample[c] = data_sample[c] + mn
    if invert_image:
        data_sample = - data_sample
    return data_sample

def gama_shift(image, c, gama):
    image /= torch.max(image)
    return  c * torch.pow(image, gama)

class UnetDataTransform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(
        self,
        mask_func: Optional[MaskFunc] = None,
        use_seed: bool = True,
        noise_aug = False,
        noise_rate = 0.15,
        use_gama = False,
        gama = 3.0
    ):
        """
        Args:
            which_challenge: Challenge from ("singlecoil", "multicoil").
            mask_func: Optional; A function that can create a mask of
                appropriate shape.
            use_seed: If true, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """

        self.mask_func = mask_func
        self.use_seed = use_seed
        self.noise_aug = noise_aug
        self.noise_rate = noise_rate

        self.use_gama = use_gama
        self.gama = gama

    def __call__(
        self,
        kspace: np.ndarray,
        fname: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data or (rows, cols) for single coil data.
            mask: Mask from the test dataset.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.

        Returns:
            tuple containing:
                image: Zero-filled input image.
                target: Target image converted to a torch.Tensor.
                mean: Mean value used for normalization.
                std: Standard deviation value used for normalization.
                fname: File name.
                slice_num: Serial number of the slice.
        """
        kspace = to_tensor(kspace)
        p = random.uniform(0, 1)
        #add noise
        if self.noise_aug and p > 0.5:
            noise_image = fastmri.ifft2c(kspace)
            max_vaule = torch.max(fastmri.complex_abs(noise_image))
            noise_image = self.rician_noise(noise_image, max_vaule)
            kspace = fastmri.fft2c(noise_image)

        p = random.uniform(0, 1)
        if self.use_gama and p > 0.5:
            image = fastmri.ifft2c(kspace)
            #image = gama_shift(image, c=1, gama=self.gama)
            image = augment_gamma(image)
            kspace = fastmri.fft2c(image)


        seed = None if not self.use_seed else tuple(map(ord, fname))
        masked_kspace, mask = apply_mask(kspace, self.mask_func, seed)

        # inverse Fourier transform to get zero filled solution
        image = fastmri.ifft2c(masked_kspace)

        # absolute value
        image = fastmri.complex_abs(image)

        # normalize input
        image, mean, std = normalize_instance(image, eps=1e-11)

        target = fastmri.ifft2c(kspace)
        target = fastmri.complex_abs(target)

        target = normalize(target, mean, std, eps=1e-11)

        return image, target, mean, std

    def rician_noise(self, X, noise_std):
        #Add rician noise with variance sampled uniformly from the range 0 and 0.1
        noise_std = random.uniform(0, noise_std*self.noise_rate)
        Ir = X + noise_std * torch.randn(X.shape)
        Ii = noise_std*torch.randn(X.shape)
        In = torch.sqrt(Ir ** 2 + Ii ** 2)
        return In


class VarNetDataTransform:
    """
    Data Transformer for training VarNet models.
    """

    def __init__(self, mask_func: Optional[MaskFunc] = None, use_seed: bool = True):
        """
        Args:
            mask_func: Optional; A function that can create a mask of
                appropriate shape. Defaults to None.
            use_seed: If True, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """
        self.mask_func = mask_func
        self.use_seed = use_seed

    def __call__(
        self,
        kspace: np.ndarray,
        mask: np.ndarray,
        target: np.ndarray,
        attrs: Dict,
        fname: str,
        slice_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, int, float, torch.Tensor]:
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data.
            mask: Mask from the test dataset.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.

        Returns:
            tuple containing:
                masked_kspace: k-space after applying sampling mask.
                mask: The applied sampling mask
                target: The target image (if applicable).
                fname: File name.
                slice_num: The slice index.
                max_value: Maximum image value.
                crop_size: The size to crop the final image.
        """
        if target is not None:
            target = to_tensor(target)
            max_value = attrs["max"]
        else:
            target = torch.tensor(0)
            max_value = 0.0

        kspace = to_tensor(kspace)
        seed = None if not self.use_seed else tuple(map(ord, fname))
        acq_start = attrs["padding_left"]
        acq_end = attrs["padding_right"]

        crop_size = torch.tensor([attrs["recon_size"][0], attrs["recon_size"][1]])

        if self.mask_func:
            masked_kspace, mask = apply_mask(
                kspace, self.mask_func, seed, (acq_start, acq_end)
            )
        else:
            masked_kspace = kspace
            shape = np.array(kspace.shape)
            num_cols = shape[-2]
            shape[:-3] = 1
            mask_shape = [1] * len(shape)
            mask_shape[-2] = num_cols
            mask = torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))
            mask = mask.reshape(*mask_shape)
            mask[:, :, :acq_start] = 0
            mask[:, :, acq_end:] = 0

        return (
            masked_kspace,
            mask.byte(),
            target,
            fname,
            slice_num,
            max_value,
            crop_size,
        )
