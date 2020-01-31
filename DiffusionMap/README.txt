This is the demo code to train spec-diff-net to learn diffusion distance on image. Please refer to the paper
[Jian Sun, Zongben Xu. Neural diffusion distance for image segmentation. NeurIPS 2019]

--Functions
'train_diffusionMap_grid_BSD_kerkmeans.py' is the function for pre-training spec-diff-net with LR kernel matching loss
'train_diffusionMap_grid_BSD_kerkmeans_full.py' is the function for training spec-diff-net using full training loss.
'eval_diffusionMap_grid_kerkmeans.py' is the function for kernel k-means for hierachical image segmentation.


--Pretrained model in folder of 'snapshots_diffMap_grid'
'BSD_diffMap120000_clus_BSD500_trainval.pth' is the pretrained network (120000 training steps) on BSD500 train+val dataset
'BSD_diffMap164000_clus_BSD500_trainval_full.pth' is fully trained network by 'train_diffusionMap_grid_BSD_kerkmeans_full.py'

To train from scratch, please change the settings of path to your own case in the functions of 'train_diffusionMap_grid_BSD_kerkmeans.py' and 'train_diffusionMap_grid_BSD_kerkmeans_full.py'.
For example, in train_diffusionMap_grid_BSD_kerkmeans.py
(1) please modify RESTORE_FROM = './initModel/MS_DeepLab_resnet_pretrained_COCO_init.pth'
(2) please modify Line 315 in main from 'if 0' to 'if 1' to load the initial model

In 'eval_diffusionMap_grid_kerkmeans.py', please change FOLDER_PSEUDO_Label to your favorite path.
