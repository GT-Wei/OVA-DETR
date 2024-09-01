# OVA-DETR

This repository contains the official implementation of [OVA-DETR](https://arxiv.org/abs/2408.12246)

**[OVA-DETR: Open Vocabulary Aerial Object Detection Using Image-Text Alignment and Fusion](https://arxiv.org/abs/2408.12246)**

[Guoting Wei](https://arxiv.org/search/cs?searchtype=author&query=Wei,+G), [Xia Yuan](https://arxiv.org/search/cs?searchtype=author&query=Yuan,+X), [Yu Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu,+Y), [Zhenhao Shang](https://arxiv.org/search/cs?searchtype=author&query=Shang,+Z), [Kelu Yao](https://arxiv.org/search/cs?searchtype=author&query=Yao,+K), [Chao Li](https://arxiv.org/search/cs?searchtype=author&query=Li,+C), [Qingsen Yan](https://arxiv.org/search/cs?searchtype=author&query=Yan,+Q), [Chunxia Zhao](https://arxiv.org/search/cs?searchtype=author&query=Zhao,+C), [Haokui Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang,+H), [Rong Xiao](https://arxiv.org/search/cs?searchtype=author&query=Xiao,+R)



## Partial results

Figure 1: ![](./images/Figure-1.jpg)

Compared OVA-DETR with recently advanced open-vocabulary detectors in terms of speed and recall. All methods are evaluated on DIOR dataset under zero shot detection. The inference speeds were measured on a 3090 GPU by default, except that DescReg was measured on a 4090 GPU



Figure 2: ![](./images/Figure-2.jpg)

Overall architecture of OVA-DETR.The improvements of OVA-DETR can be summarized into two main components: the Image-Text Alignment and the Bidirectional Vision-Language Fusion.



Figure 5:![](./images/Figure-5.jpg) 

Qualitative results for zero-shot detection on the xView,DIOR,and DOTA datasets, focusing on novel classes.The green rectangles represent predicted bounding boxes, while red rectangles denote ground truth bounding boxes.













 