# Efficient Track Anything
[[`📕Project`](https://yformer.github.io/efficient-track-anything/)][[`🤗Gradio Demo`](https://10f00f01361a8328a4.gradio.live)][[`📕Paper`](https://arxiv.org/pdf/2411.18933)][[`🤗Checkpoints`]](https://huggingface.co/yunyangx/efficient-track-anything/tree/main)

Implementation of real-time EfficientTAM

The **Efficient Track Anything Model(EfficientTAM)** takes a vanilla lightweight ViT image encoder. An efficient memory cross-attention is proposed to further improve the efficiency. Our EfficientTAMs are trained on SA-1B (image) and SA-V (video) datasets. EfficientTAM achieves comparable performance with SAM 2 with improved efficiency. Our EfficientTAM can run **>10 frames per second** with reasonable video segmentation performance on **iPhone 15**. Try our demo with a family of EfficientTAMs at [[`🤗Gradio Demo`](https://10f00f01361a8328a4.gradio.live)].


## Model
EfficientTAM checkpoints are available at the [Hugging Face Space](https://huggingface.co/yunyangx/efficient-track-anything/tree/main).

## Getting Started

### 1. Installation

```bash
git clone https://github.com/yformer/EfficientTAM.git
cd EfficientTAM
conda create -n efficient_track_anything python=3.12
conda activate efficient_track_anything
pip install -e .
```
### 2. Download Checkpoints

```bash
cd checkpoints
./download_checkpoints.sh
```
### 3. Run tam_app.py inside notebooks folder

## License
Efficient track anything checkpoints and codebase are licensed under [Apache 2.0](./LICENSE).

## Acknowledgement

+ [SAM2](https://github.com/facebookresearch/sam2)
+ [SAM2-Video-Predictor](https://huggingface.co/spaces/fffiloni/SAM2-Video-Predictor)
+ [florence-sam](https://huggingface.co/spaces/SkalskiP/florence-sam)
+ [SAM](https://github.com/facebookresearch/segment-anything)
+ [EfficientSAM](https://github.com/yformer/EfficientSAM)

If you're using Efficient Track Anything in your research or applications, please cite using this BibTeX:
```bibtex


@article{xiong2024efficienttam,
  title={Efficient Track Anything},
  author={Yunyang Xiong, Chong Zhou, Xiaoyu Xiang, Lemeng Wu, Chenchen Zhu, Zechun Liu, Saksham Suri, Balakrishnan Varadarajan, Ramya Akula, Forrest Iandola, Raghuraman Krishnamoorthi, Bilge Soran, Vikas Chandra},
  journal={preprint arXiv:2411.18933},
  year={2024}
}
```
