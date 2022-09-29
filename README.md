# Global-Local-Attention-Module-Pytorch
Unoffical Implementation of the Global Local Attention Module (GLAM) in PyTorch

![image](https://user-images.githubusercontent.com/79294502/192976117-67fa4a17-eec0-4dda-987d-3c1fc2ffe554.png)

## Paper

Song, C. H., Han, H. J., & Avrithis, Y. (2022). All the attention you need: Global-local, spatial-channel attention for image retrieval. In *Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision* (pp. 2754-2763).

## Usage

```python
glam = GLAM(in_channels=32, num_reduced_channels=16, feature_map_size=8, kernel_size=5)

feature_maps = torch.randn(16, 32, 8, 8)

glam(feature_maps) # shape (16, 32, 8, 8)
```
