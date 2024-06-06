from typing import Tuple, List, Optional, Union
import numpy as np
import torch
from matplotlib import cm, pyplot as plt
from PIL import Image
import numpy as np
from matplotlib import pylab as P
import cv2
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

'''
This code contains some interpretable code. You can use ShowGrayscaleImage to visualize the gradient. 
Note that in order to better visualize the original image and the images after RPM and ARN, we have squared the gradient for visualization.
'''
def overlay_mask(img: Image.Image, mask: Image.Image, colormap: str = "jet", alpha: float = 0.5) -> Image.Image:
    """Overlay a colormapped mask on a background image
    >>> from PIL import Image
    >>> import matplotlib.pyplot as plt
    >>> img = ...
    >>> cam = ...
    >>> overlay = overlay_mask(img, cam)
    Args:
        img: background image
        mask: mask to be overlayed in grayscale
        colormap: colormap to be applied on the mask
        alpha: transparency of the background image
    Returns:
        overlayed image
    Raises:
        TypeError: when the arguments have invalid types
        ValueError: when the alpha argument has an incorrect value
    """
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    if not isinstance(img, Image.Image) or not isinstance(mask, Image.Image):
        raise TypeError("img and mask arguments need to be PIL.Image")
    if not isinstance(alpha, float) or alpha < 0 or alpha >= 1:
        raise ValueError("alpha argument is expected to be of type float between 0 and 1")
    cmap = cm.get_cmap(colormap)
    # Resize mask and apply colormap
    overlay = mask.resize(img.size, resample=Image.BICUBIC)
    overlay = (255 * cmap(np.asarray(overlay) ** 2)[:, :, :3]).astype(np.uint8)
    # Overlay the image with the mask
    overlayed_img = Image.fromarray((alpha * np.asarray(img) + (1 - alpha) * overlay).astype(np.uint8))
    plt.imshow(overlayed_img);
    plt.axis('off');
    plt.tight_layout();
    plt.savefig("EIT.png")
    return overlayed_img


def VisualizeImageGrayscale(image_3d, percentile=99):
    r"""Returns a 3D tensor as a grayscale 2D tensor.

    This method sums a 3D tensor across the absolute value of axis=2, and then
    clips values at a given percentile.
    """
    image_2d = np.sum(np.abs(image_3d), axis=2)
    vmax = np.percentile(image_2d, percentile)
    vmin = np.min(image_2d)
    return np.clip((image_2d - vmin) / (vmax - vmin), 0, 1)


def VisualizeImageDiverging(image_3d, percentile=99):
    r"""Returns a 3D tensor as a 2D tensor with positive and negative values.
    """
    image_2d = np.sum(image_3d, axis=2)
    span = abs(np.percentile(image_2d, percentile))
    vmin = -span
    vmax = span
    return np.clip((image_2d - vmin) / (vmax - vmin), -1, 1)


def ShowGrayscaleImage(image_3d, title='', ax=None):
    """
    example:
     >>> test = ShowGrayscaleImage(image_3d=..., title= "", ax = ...)
    """
    if isinstance(image_3d, torch.Tensor):
        image_3d = image_3d.clone().detach().cpu().squeeze().numpy().transpose(1, 2, 0)
    im = VisualizeImageGrayscale(image_3d)
    if ax is None:
        P.figure()
    P.axis('off')
    P.imshow(im, cmap=P.cm.gray, vmin=0, vmax=1)
    P.title(title)
    P.savefig(f"{title}.png", dpi=750)
    P.show()


def ShowColorscaleImage(
        images: List[Union[np.ndarray, torch.Tensor]],
        titles: List[str],
        output_path: str = None,
        save: bool = False
):
    """
    example:
    >>> ShowColorscaleImage(([images], [title]))
    """
    for i in range(len(images)):
        if isinstance(images[i], torch.Tensor):
            images[i] = images[i].clone().detach().cpu().squeeze().numpy().transpose(1, 2, 0)
            images[i] = np.fabs(images[i]) / np.max(images[i])
    fig, axs = plt.subplots(1, len(images))
    fig.set_figheight(10)
    fig.set_figwidth(16)

    for i, (title, img) in enumerate(zip(titles, images)):
        plt.imshow(img)
        plt.title(title, fontsize='large')
        plt.axis("off")
    if save is True:
        plt.savefig(output_path, dpi=750)
    plt.show()
    fig.tight_layout()


def save_saliency_map(image, saliency_map, filename):
    """
    Save saliency map on image.
    Args:
        image: Tensor of size (3,H,W)
        saliency_map: Tensor of size (1,H,W)
        filename: string with complete path and file extension
    """
    saliency_map = torch.abs(saliency_map.squeeze_()).sum(0, keepdim=True)
    image = image.squeeze().data.cpu().numpy()
    saliency_map = saliency_map.data.cpu().numpy()
    saliency_map = saliency_map - saliency_map.min()
    saliency_map = saliency_map / saliency_map.max()
    saliency_map = saliency_map.clip(0, 1)
    saliency_map = np.uint8(saliency_map * 255).transpose(1, 2, 0)
    saliency_map = cv2.resize(saliency_map, (224, 224))
    image = np.uint8(image * 255).transpose(1, 2, 0)
    image = cv2.resize(image, (224, 224))
    # Apply JET colormap
    color_heatmap = cv2.applyColorMap(saliency_map, cv2.COLORMAP_JET)
    # Combine image with heatmap
    img_with_heatmap = np.float32(color_heatmap) + np.float32(image)
    img_with_heatmap = img_with_heatmap / np.max(img_with_heatmap)
    cv2.imwrite(filename, np.uint8(255 * img_with_heatmap))
