import matplotlib.pyplot as plt
from typing import Optional, List


# Viz.
def plot_sample(
    ax,
    item,
    annotate: Optional[bool] = False,
    is_transformed: Optional[bool] = False
):
    """
    Plot a specified samples.
    
    Parameters
    ----------
    ax
        A dataframe with the metadata on the samples.
    item
        A pytorch Dataset instance.
    annotate
        Optionally, whether to annotate the samples or not.
    is_transformed
        Optionally, whether the dataset has been transformed 
        specifically to a pytorch tensor.
    """
    # Show image.
    if is_transformed:
        ax.imshow(np.transpose(item['image'], [1, 2, 0]))
    else:
        ax.imshow(item['image'])
    
    # Add annotations if necessary.
    if annotate:
        ax.imshow(item['mask'],cmap='coolwarm', alpha=0.3)

    ax.axis("off")
    # 0: Organ, 4: Sex
    ax.set_title(f"{item['metadata'].organ}, {item['metadata'].sex}") 


def plot_samples(
    dataset,
    indices: list,
    annotate: bool = False,
    cols: int = 2,
    figsize: tuple = (18, 5),
    is_transformed: Optional[bool] = False
):
    """
    Plot a grid of specified samples.
    
    Parameters
    ----------
    hhhhb_dataset
        A pytorch Dataset object.
    indices
        A list of the item indices to plot.
    annotate
        Optionally, whether to annotate the samples or not.
    cols
        Optionally, the number of columns for the grid.
    figsize
        Optionally, a tuple containg the figure size in the form (width, height).
    is_transformed
        Optionally, whether the dataset has been transformed 
        specifically to a pytorch tensor.
    """
    # Infer rows.
    items = len(indices)
    rows, x = items // cols, items % cols # x carries the reminder if it exists.
    rows = rows if x == 0 else rows + 1
    
    # Set up figure.
    if not (isinstance(figsize, tuple) and len(figsize)==2):
        raise ValueError("Please provide a valid figsize i.e. (width, height)")
    fig = plt.figure(figsize=(figsize[0], figsize[1] * rows))
    
    # Plotting loop.
    for idx, index in enumerate(indices):
        ax = fig.add_subplot(rows, cols, idx + 1)
        plot_sample(ax, dataset[index], annotate, is_transformed)
        
    plt.tight_layout()