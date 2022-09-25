import pandas as pd
import tifffile as tiff
from torch.utils.data import Dataset
from .utils import rle2mask

from typing import Optional, List

# The dataset.
class HHHHBDataset(Dataset):
    """
    Reads in images, transfroms pixel values and serves a dictionary containing the image ids,
    image tensors and the label masks.
    """
    def __init__(
        self,
        data: pd.DataFrame,
        transforms: Optional[list] = None,
        metadata: Optional[bool] = False
    ):
        """
        Instantiate the HHHHBDataset.
        
        Parameters
        ----------
        data
            A dataframe with a row for each biopsy image.
        transforms
            Optionally, a list of transforms to apply to the feature data. (Augmentations).
        metadata
            Optionally, add metadata to the returned dictionary, necessary for plotting and evaluation
            but not model training.
        """
        self.data = data
        self.transforms = transforms
        self.metadata = metadata
        
    def __len__(self):
        return self.data['id'].nunique()
    
    def __getitem__(
        self, 
        idx: int
    ):
        # Loads an n-channel image from a chip-level dataframe.
        img_metadata = self.data.loc[idx]
        
        # Read in the image.
        img_arr = tiff.imread(img_metadata.file_path)
        
        # Load mask.
        mask_arr = rle2mask(img_metadata.rle, (img_metadata.img_width, img_metadata.img_height))
        
        # Apply data augmentations, if provided.
        if self.transforms:
            augmented = self.transforms(image=img_arr, mask=mask_arr)
            # Get augmentations.
            img_arr = augmented['image']
            mask_arr = augmented['mask']
        
        # Prepare the dictionary for item.
        item = {
            "id": img_metadata["id"], 
            "image": img_arr, # Change from HxWxC to CxHxW for pytorch. ToTensorV2
            "mask": mask_arr
        }
        
        # Add metadata.
        if self.metadata:
            item["metadata"] = img_metadata[['organ', 'pixel_size', 'tissue_thickness', 'age', 'sex']]
        return item