#!/usr/bin/env bash

#### pre-train the diffusion network using only LR kernel matching loss
/usr/bin/python3.5 /usr/bin/python3.5 /home/jiansun/Projects/DiffusionMap/train_diffusionMap_grid_BSD_kerkmeans.py

#### Further train using full training loss on full image, batch size = 1 to handle images in different resolution
/usr/bin/python3.5 /usr/bin/python3.5 /home/jiansun/Projects/DiffusionMap/train_diffusionMap_grid_BSD_kerkmeans_fullimage.py  ##

#### Evaluate for hierachical semantic segmentation
/usr/bin/python3.5 /home/jiansun/Projects/DiffusionMap/eval_diffusionMap_grid_kerkmeans.py

