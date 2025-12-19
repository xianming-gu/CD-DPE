# [AAAI 2026] CD-DPE: Dual-Prompt Expert Network based on Convolutional Dictionary Feature Decoupling for Multi-Contrast MRI Super-Resolution

Xianming Gu<sup>1</sup>, Lihui Wang<sup>1,∗</sup>, Ying Cao<sup>1</sup>, Zeyu Deng<sup>1</sup>, Yingfeng Ou<sup>a</sup>, Guodong Hu<sup>a</sup> and Yi Chen<sup>1,2</sup>

<sup>1</sup> Key Laboratory of Advanced Medical Imaging and Intelligent Computing of Guizhou Province,
Engineering Research Center of Text Computing & Cognitive Intelligence, Ministry of Education,
College of Computer Science and Technology, Guizhou University, Guiyang, China

<sup>2</sup> The D-Lab, Department of Precision Medicine, GROW-School for Oncology and Reproduction,
Maastricht University, 6200 MD Maastricht, the Netherlands

<sup>*</sup> Corresponding author.

-*[Paper]*: [[arxiv]](https://arxiv.org/abs/2511.14014), [AAAI-TBD]

-*[Email]*: [xianming_gu@foxmail.com](mailto:xianming_gu@foxmail.com) (Xianming Gu)

-[*[Supplementary Materials]*](https://github.com/xianming-gu/CD-DPE/blob/main/Supplementary-material/Supplementary-material.pdf)


## Citation

```
TBD
```

## Abstract

Multi-contrast magnetic resonance imaging (MRI) super-resolution intends to reconstruct high-resolution (HR) images from low-resolution (LR) scans by leveraging structural information present in HR reference images acquired with different contrasts. This technique enhances anatomical detail and soft tissue differentiation, which is vital for early diagnosis and clinical decision-making. However, inherent contrasts disparities between modalities pose fundamental challenges in effectively utilizing reference image textures to guide target image reconstruction, often resulting in suboptimal feature integration. To address this issue, we propose a dual-prompt expert network based on a convolutional dictionary feature decoupling (CD-DPE) strategy for multi-contrast MRI super-resolution. Specifically, we introduce an iterative convolutional dictionary feature decoupling module (CD-FDM) to separate features into cross-contrast and intra-contrast components, thereby reducing redundancy and interference. To fully integrate these features, a novel dual-prompt feature fusion expert module (DP-FFEM) is proposed. This module uses a frequency prompt to guide the selection of relevant reference features for incorporation into the target image, while an adaptive routing prompt determines the optimal method for fusing reference and target features to enhance reconstruction quality. Extensive experiments on public multi-contrast MRI datasets demonstrate that CD-DPE outperforms state-of-the-art methods in reconstructing fine details. Additionally, experiments on unseen datasets demonstrated that CD-DPE exhibits strong generalization capabilities.


