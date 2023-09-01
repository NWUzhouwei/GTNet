# GTNet
## Requirements
- Python >= 3.7
- PyTorch >= 1.2
- CUDA >= 10.0
- Package: glob, h5py, sklearn, plyfile,tensorboardX
#### Overall Network architecture:
The network architecture of GTNet is shown below:

<p float="left">
    <img src="image/network.png"/>
</p>

#### Implementation details
- GTNet is mainly composed of Graph Transformer, which is mainly divided into Local Transformer and Global Transformer:

<p float="left">
    <img src="image/lg.png"/>
</p>

- Dynamic graph update process:

<p float="left">
    <img src="image/graph.png"/>
</p>

## Point Cloud Classification
### Run the training script:
python main_cls.py
### Run the evaluation script after training finished:
python main_cls.py --exp_name=cls_eval --eval=True --model_path='xxx'
### Performance:
ModelNet40 dataset
|  | Mean Class Acc | Overall Acc |
| :---: | :---: | :---: |
| This repo (2048 points) | **91.2** | **93.6** |

&nbsp;

## Point Cloud Part Segmentation
**Note:** The training modes **'full dataset'** and **'with class choice'** are different.

### Run the training script:
python main_partseg.py --exp_name=partseg
- With class choice, for example Earphone
python main_partseg.py --exp_name=partseg_airplane --class_choice=Earphone

### Run the evaluation script after training finished:
python main_partseg.py --exp_name=partseg_eval --eval=True --model_path='xxx'


### Performance:
ShapeNet part dataset

| | Mean IoU | Airplane | Bag | Cap | Car | Chair | Earphone | Guitar | Knife | Lamp | Laptop | Motor | Mug | Pistol | Rocket | Skateboard | Table
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
| This repo | 85.1 | 84.1 | 77.7 | 82.7 | 77.4 | 91.0 | 76.3 | 91.8 | 86.5 | 83.5 | 96.1 | 58.5 | 92.4 | 81.9 | 53.5 | 76.6 | 82.9 |


### Visualization:
#### Usage:
Both .txt and .ply file can be loaded into [MeshLab](https://www.meshlab.net) for visualization. For the usage of MeshLab on .txt file. The .ply file can be directly loaded into MeshLab by dragging.
python main_partseg.py --exp_name=partseg_eval --eval=True --model_path='xxx' --visu=Earphone_0 --visu_format=ply
#### Results:
The visualization result:

<p float="left">
    <img src="image/partseg_visu.png"/>
</p>

&nbsp;
## Point Cloud Semantic Segmentation on the S3DIS Dataset
You have to download `Stanford3dDataset_v1.2_Aligned_Version.zip` manually from https://goo.gl/forms/4SoGp4KtH1jfRqEj2 and place it under `data/`
### Run the training script:

This task uses 6-fold training, such that 6 models are trained leaving 1 of 6 areas as the testing area for each model. 
python main_semseg.py --test_area=1
python main_semseg.py --test_area=2
python main_semseg.py --test_area=3
...
python main_semseg.py --test_area=6
### Run the evaluation script after training finished:
- Evaluate in all areas after 6 models are trained
python main_semseg.py --exp_name=semseg_eval --test_area=all --eval=True --model_root='xxx'

### Performance:
Stanford Large-Scale 3D Indoor Spaces Dataset (S3DIS) dataset

|  | Mean IoU | Overall Acc |
| :---: | :---: | :---: |
| This repo | **64.3** | **86.6** |
### Visualization: 
#### Usage:
Both .txt and .ply file can be loaded into [MeshLab](https://www.meshlab.net) for visualization. For the usage of MeshLab on .txt file. The .ply file can be directly loaded into MeshLab by dragging.
python main_semseg_s3dis.py --exp_name=semseg_eval --test_area=all --eval=True --model_root='xxx' --visu_format=ply
#### Results:
The visualization result:

<p float="left">
    <img src="image/semseg_visu.png"/>
</p>

Color map:
<p float="left">
    <img src="image/semseg_colors.png" width="800"/>
</p>


## Citation
GTNet code refers to the following codes: 

[DGCNN](https://github.com/antao97/dgcnn.pytorch)

[Point Transformer](https://github.com/qq456cvb/Point-Transformers)
