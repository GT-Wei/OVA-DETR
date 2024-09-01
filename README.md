<div style="text-align: center;">
  <h1 style="font-size: 48px; margin: 0;">OVA-DETR</h1>
  <p style="font-size: 24px; margin: 0;">Open Vocabulary Aerial Object Detection Using Image-Text Alignment and Fusion</p>
</div>

<div align="center">
<br>
<a href="https://arxiv.org/search/cs?searchtype=author&query=Wei,+G">Guoting Wei</a><sup><span>1,4,*</span></sup>, 
<a href="https://arxiv.org/search/cs?searchtype=author&query=Yuan,+X">Xia Yuan</a><sup><span>1,*</span></sup>,
<a href="https://arxiv.org/search/cs?searchtype=author&query=Liu,+Y">Yu Liu</a><sup><span>3,🌟</span></sup>,
<a href="https://arxiv.org/search/cs?searchtype=author&query=Shang,+Z"> Zhenhao Shangu</a><sup><span>2</span></sup>,
<a href="https://arxiv.org/search/cs?searchtype=author&query=Yao,+K">Kelu Yao</a><sup><span>3</span></sup>,
<a href="https://arxiv.org/search/cs?searchtype=author&query=Li,+C">Chao Li</a><sup><span>3</span></sup>
<a href="https://arxiv.org/search/cs?searchtype=author&query=Yan,+Q">Qingsen Yan</a><sup><span>2</span></sup>
<a href="https://arxiv.org/search/cs?searchtype=author&query=Zhao,+C">Chunxia Zhao</a><sup><span>1</span></sup>
<a href="https://arxiv.org/search/cs?searchtype=author&query=Zhang,+H">Haokui Zhang</a><sup><span>2,🌟,4</span></sup>
<a href="https://arxiv.org/search/cs?searchtype=author&query=Xiao,+R">Rong Xiao</a><sup><span>4</span></sup>
</br>

\* Equal contribution 🌟 Project lead 📧 Corresponding author

<sup>1</sup> Nanjing University of Science and Technology,  <sup>3 </sup>Zhejiang Lab

<sup>2</sup> Northwestern Polytechnical University, <sup>4</sup>Intellifusion



This repository contains the official implementation of [OVA-DETR](https://arxiv.org/abs/2408.12246)

**[OVA-DETR: Open Vocabulary Aerial Object Detection Using Image-Text Alignment and Fusion](https://arxiv.org/abs/2408.12246)**



## Partial results

Figure 1: ![](./images/Figure-1.jpg)

Compared OVA-DETR with recently advanced open-vocabulary detectors in terms of speed and recall. All methods are evaluated on DIOR dataset under zero shot detection. The inference speeds were measured on a 3090 GPU by default, except that DescReg was measured on a 4090 GPU



Figure 2: ![](./images/Figure-2.jpg)

Overall architecture of OVA-DETR.The improvements of OVA-DETR can be summarized into two main components: the Image-Text Alignment and the Bidirectional Vision-Language Fusion.



Figure 5:![](./images/Figure-5.jpg) 

Qualitative results for zero-shot detection on the xView,DIOR,and DOTA datasets, focusing on novel classes.The green rectangles represent predicted bounding boxes, while red rectangles denote ground truth bounding boxes.













 
