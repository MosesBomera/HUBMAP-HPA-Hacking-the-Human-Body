"""Run length encoding utilities."""

import numpy as np

def mask2rle(
    mask: np.array
):
    """
    Encode mask using run length encoding.
    
    Parameters
    ----------
    mask
         A numpy array, 1 - mask, 0 - background
    
    Returns
    -------
    A run length encoding of the mask as a string formated.
    
    Notes
    -----
    Reference: https://www.kaggle.com/paulorzp/rle-functions-run-length-encode-decode
    """
    pixels= mask.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle2mask(
    mask_rle: str, 
    shape: tuple = (3000,3000)
):
    """
    Convert a run length encoding to a mask.
    
    Parameters
    ----------
    mask_rle
        A run-length as string formated (start length).
    shape
        The shape of the image to which the rle is to be unencoded (width,height).
        
    Returns
    -------
        A numpy array, 1 - mask, 0 - background.
        
    Notes
    -----
    Reference: https://www.kaggle.com/paulorzp/rle-functions-run-length-encode-decode
    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T