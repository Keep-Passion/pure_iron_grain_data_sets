import random
import numpy as np
import numba
from skimage import morphology
from skimage.color import label2rgb
import matplotlib.pyplot as plt
import h5py


@numba.jit(nopython=True, nogil=True)
def get_boundary_from_label(labeled, neighbors=4):
    """
    Get boundary map from labeled image
    Each pixel have its own label. If one pixel find different label in its neighborhood,
    the pixel should be boundary pixel and set to 255. Otherwise, it should be set to 0.
    从标注图像中获得边缘图像
    每个像素均有其自己的label,如果在一个像素的领域内找到与其不同的label,
    说明该像素是边缘像素，应该置为255，否则，应被置为0
    :param labeled: np.array
        The labeled image.标注图像
    :param neighbors: {4, 8}, int, optional
        Default 4 neighbors for pixel neighborhood.连通区域类型，默认为4邻接
    :return: edges, np.array
        The edge map for labeled.返回标注图对应的边缘图像
    """
    boundary = np.zeros(labeled.shape, dtype=np.uint8)
    for row in range(0, labeled.shape[0]):
        for col in range(0, labeled.shape[1]):
            value = labeled[row, col]
            is_boundary = False
            if value == 0:
                boundary[row, col] = 255
                continue
            if neighbors == 4:
                if row != 0 and value != labeled[row-1, col]:
                    is_boundary = True
                elif col != 0 and value != labeled[row, col-1]:
                    is_boundary = True
                elif row != labeled.shape[0]-1 and value != labeled[row+1, col]:
                    is_boundary = True
                elif col != labeled.shape[1]-1 and value != labeled[row, col+1]:
                    is_boundary = True
            elif neighbors == 8:
                if row != 0 and value != labeled[row-1, col]:
                    is_boundary = True
                elif row != 0 and col != 0 and value != labeled[row-1, col-1]:
                    is_boundary = True
                elif row != 0 and col != labeled.shape[1]-1 and value != labeled[row-1, col+1]:
                    is_boundary = True
                elif row != labeled.shape[0]-1 and value != labeled[row+1, col]:
                    is_boundary = True
                elif row != labeled.shape[0]-1 and col != 0 and value != labeled[row+1, col-1]:
                    is_boundary = True
                elif row != labeled.shape[0]-1 and col != labeled.shape[1]-1 and value != labeled[row+1, col+1]:
                    is_boundary = True
                elif col != 0 and value != labeled[row, col-1]:
                    is_boundary = True
                elif col != labeled.shape[1]-1 and value != labeled[row, col+1]:
                    is_boundary = True
            if is_boundary:
                boundary[row, col] = 255
    return boundary


file_h5 = h5py.File("pure_iron_grain_data_sets.hdf5", "r")
print("Visualize the structure of file_h5")
file_h5.visit(print)

real_group = file_h5

r_select_index = random.randint(0, 1)  # random select an image to show 随机选择一层展示
r_origin_image = real_group["image"][:, :, r_select_index]

r_label_image = real_group["label"][:, :, r_select_index]
r_label_color_image = label2rgb(r_label_image)

r_boundary_image = real_group["boundary"][:, :, r_select_index]


show_length = 200
plt.figure(figsize=(30, 30))
plt.subplot(131)
plt.imshow(r_origin_image[0:show_length, 0:show_length], cmap="gray")
plt.axis('off')
plt.title("real image", fontsize=60)
plt.subplot(132)
plt.imshow(r_label_color_image[0:show_length, 0:show_length])
plt.axis('off')
plt.title("real label", fontsize=60)
plt.subplot(133)
plt.imshow(r_boundary_image[0:show_length, 0:show_length], cmap="gray")
plt.axis('off')
plt.title("real boundary", fontsize=60)

plt.show()
