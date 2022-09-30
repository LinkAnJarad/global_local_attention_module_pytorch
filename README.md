# Global-Local-Attention-Module-Pytorch
Unoffical Implementation of the [Global Local Attention Module (GLAM)](https://arxiv.org/pdf/2107.08000.pdf) in PyTorch

![image](https://user-images.githubusercontent.com/79294502/192976117-67fa4a17-eec0-4dda-987d-3c1fc2ffe554.png)

## [Paper](https://arxiv.org/pdf/2107.08000.pdf)

Song, C. H., Han, H. J., & Avrithis, Y. (2022). All the attention you need: Global-local, spatial-channel attention for image retrieval. In *Proceedings of the IEEE/CVF  Winter Conference on Applications of Computer Vision* (pp. 2754-2763).

## Usage

```python
from global_local_attention_module_pytorch import GLAM

feature_maps = torch.randn(16, 32, 8, 8) # shape (batch_size, num_channels, height, width)
glam = GLAM(in_channels=32, num_reduced_channels=16, feature_map_size=8, kernel_size=5)

glam(feature_maps) # shape (16, 32, 8, 8), same as input
```
Note: *The height/width of the input feature maps must be at least 7, due to the 7x7 convolution (3x3 dilated conv) in the module.*
## Arguments

* `in_channels (int)`: number of channels of the input feature map
* `num_reduced_channels (int)`: number of channels that the local and global spatial attention modules will reduce the input feature map. Refer to figures 3 and 5 in the paper.
* `feaure_map_size (int)`: height/width of the feature map
* `kernel_size (int)`: scope of the inter-channel attention
