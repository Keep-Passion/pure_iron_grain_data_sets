# Pure Iron Grain Data Sets
Microstructure is of great importance for controlling the properties and performance in materials science. Among all the materials, polycrystalline structure are commonly used and studied in practice.

Recently progress in material microscopic image segmentation has been driven by high-capacity models trained on large data sets. However, unlike public data in nature and biological scenes, the produce and label of image data is very time-consuming. Because the opacity of materials, scientist can only use serial section method to obtain the 2D images of materials. Besides, the image may suffer many flaws during sample preparation, which make the labeling process consume much more times than other image data. In total, we think the progress of material microscopic image processing is hindered by the lack of public data.

Therefore, we public our data sets with its label in order to provide a referenced data sets for computer vision community.

We provide pure iron data and its corresponding labels to facilitate the researcher.

The region of interest is grain boundary, the algorithm should detect single-pixel width and closed grain boundary.
 The final goal of 2D boundary detection is to reconstruct 3D information of microstructure. And therefore find the association between structure and material properties.  

## Data preparation

The specimen was intercepted from a hot-rolled iron slab and forged into round bars with a diameter equals to 30 mm. The pure iron bars were then fully recrystallized by annealing at 880 °C for 3 h to gain uniform grain microstructures. The samples were polished for a fixed time, and each polished layer was etched with 4vol% nital solution in preparation for optical microscopy. Images of microstructure were collected by an optical microscope, and a total of 296 serial sections with an average section thickness of 1.8 μm were obtained. We used microhardness tester to produce a sets of points to ensure that the images of the same area of interest were collected.

Unlike public data in nature and biological scenes, the material microscopic image often suffer many flaws during sample preparation. As shown in Figure, (a) is the stack of serial sections for polycrystalline iron. (b) is one slice of metallographic image, it contains grain boundaries (straight and thick arrows), vague or missing boundaries (straight and thin arrows), noise (curved arrows) and spurious scratches (notched arrows).(c) is labeling result of (b) and (d) is boundary detection result of (b).

The reasons of those flaws:  
* `Vague or missing boundary`: caused by incomplete etching in 4vol% nital solution. The boundary could be recovered by information of neighbor slices.  
* `Noise`: caused in sample preparation.  
* `Spurious scratches`: unavoidably caused in polished process, which is similar to boundary and introduce difficulty in image processing.

The boundary detection task will unavoidably be hindered by those flaws.

The real image data set consists of 296 images, with resolution of 1024 × 1024.

![](./explain_image/polycrystalline_iron.jpg)

## Usage of data
All the images stored in hdf5 data type. The architecture of the data is as follow:  
```Python
pure_iron_grain_data_sets.hdf5
    image
    label
    boundary
```

We provide python code to visualize them:  
[inspect_data](https://github.com/Keep-Passion/pure_iron_grain_data_sets/blob/master/inspect_data.py): load and visualize data.

## Downloading of data
The dataset can be downloaded at [Baidu Pan](https://pan.baidu.com/s/1l6EgQcVU_mSPJ37_satVog?pwd=573s), with keys 573s.
