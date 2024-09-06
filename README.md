<h1 align="center">OVA-DETR</h1>

<p>
This repository contains the official implementation of <a href="https://arxiv.org/abs/2408.12246">OVA-DETR</a>
</p>

<h2 align="center"><a href="https://arxiv.org/abs/2408.12246">OVA-DETR: Open Vocabulary Aerial Object Detection Using Image-Text Alignment and Fusion</a></h2>

<p align="center">
<a href="https://arxiv.org/search/cs?searchtype=author&query=Wei,+G">Guoting Wei</a><sup>1,4,*</sup>, 
<a href="https://arxiv.org/search/cs?searchtype=author&query=Yuan,+X">Xia Yuan</a><sup>1,*</sup>,
<a href="https://arxiv.org/search/cs?searchtype=author&query=Liu,+Y">Yu Liu</a><sup>3,🌟</sup>,
<a href="https://arxiv.org/search/cs?searchtype=author&query=Shang,+Z">Zhenhao Shang</a><sup>2</sup>,
<a href="https://arxiv.org/search/cs?searchtype=author&query=Yao,+K">Kelu Yao</a><sup>3</sup>,
<a href="https://arxiv.org/search/cs?searchtype=author&query=Li,+C">Chao Li</a><sup>3</sup>
<a href="https://arxiv.org/search/cs?searchtype=author&query=Yan,+Q">Qingsen Yan</a><sup>2</sup>,
<a href="https://arxiv.org/search/cs?searchtype=author&query=Zhao,+C">Chunxia Zhao</a><sup>1</sup>
<a href="https://arxiv.org/search/cs?searchtype=author&query=Zhang,+H">Haokui Zhang</a><sup>2,🌟,4</sup>
<a href="https://arxiv.org/search/cs?searchtype=author&query=Xiao,+R">Rong Xiao</a><sup>4</sup>
</p>

<p align="center">
* Equal contribution 🌟 Project lead 📧 Corresponding author
</p>

<p align="center">
<sup>1</sup> Nanjing University of Science and Technology, <sup>3</sup> Zhejiang Lab<br>
<sup>2</sup> Northwestern Polytechnical University, <sup>4</sup> Intellifusion<br><br>
</p>
       

## Partial results

Figure 1: ![](./images/Figure-1.jpg)

Compared OVA-DETR with recently advanced open-vocabulary detectors in terms of speed and recall. All methods are evaluated on DIOR dataset under zero shot detection. The inference speeds were measured on a 3090 GPU by default, except that DescReg was measured on a 4090 GPU



Figure 2: ![](./images/Figure-2.jpg)

Overall architecture of OVA-DETR.The improvements of OVA-DETR can be summarized into two main components: the Image-Text Alignment and the Bidirectional Vision-Language Fusion.



Figure 5:![](./images/Figure-5.jpg) 

Qualitative results for zero-shot detection on the xView,DIOR,and DOTA datasets, focusing on novel classes.The green rectangles represent predicted bounding boxes, while red rectangles denote ground truth bounding boxes.


## Installation
1. Clone the OVA-DETR repository.
```
git clone https://github.com/GT-Wei/OVA-DETR.git
```
2. Clone the mmdetection repository (include RT-DETR cfw)
```
git clone https://github.com/flytocc/mmdetection.git
cp -r OVA-DETR/* ./mmdetection/
```
3. OVA-DETR is developed based on `torch==1.11.0+cu11.3` and `mmdetection==3.3.0`
```
conda create -n OVA-DETR python==3.8 -y
conda activate OVA-DETR

conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch

pip install -U openmim
mim install mmengine
mim install "mmcv==2.0.0"
pip install transformers open_clip_torch
pip install git+https://github.com/openai/CLIP.git

cd mmdetection 
pip install -v -e .

mkdir pretrain_model
wget https://github.com/flytocc/mmdetection/releases/download/model_zoo/rtdetr_r50vd_8xb2-72e_coco_ff87da1a.pth
wget https://github.com/GT-Wei/OVA-DETR/releases/download/v1.0.0/epoch_30.pth
wget https://github.com/GT-Wei/OVA-DETR/releases/download/v1.0.0/epoch_45.pth
```
4. Training
```
eg: CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh configs/OVA_DETR/OVA_DETR_4xb4-80e_dior_dota_xview.py 4
```
5. Evaluation
```
eg: CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_test.sh configs/OVA_DETR/OVA_DETR_4xb4-80e_dior_dota_xview.py ./pretrain_model/epoch30.pt 4
```

# Acknowledgement
We are grateful to the contributors for their crucial integration of RT-DETR into the mmdetection framework. We implemented OVA-DETR based on their shared resources available at [mmdetection](https://github.com/flytocc/mmdetection).

```
@article{wei2024ova,
  title={OVA-DETR: Open Vocabulary Aerial Object Detection Using Image-Text Alignment and Fusion},
  author={Wei, Guoting and Yuan, Xia and Liu, Yu and Shang, Zhenhao and Yao, Kelu and Li, Chao and Yan, Qingsen and Zhao, Chunxia and Zhang, Haokui and Xiao, Rong},
  journal={arXiv preprint arXiv:2408.12246},
  year={2024}
}
```




 
