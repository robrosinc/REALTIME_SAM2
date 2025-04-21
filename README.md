# Real Time Segment Anything 2
**Implementation of real-time EfficientTAM**

We have enabled real-time addition of objects to track during tracking. 

The Efficient Track Anything Model(EfficientTAM) takes a vanilla lightweight ViT image encoder. An efficient memory cross-attention is proposed to further improve the efficiency. EfficientTAMs are trained on SA-1B (image) and SA-V (video) datasets. EfficientTAM achieves comparable performance with SAM 2 with improved efficiency.

[[`📕Project`](https://yformer.github.io/efficient-track-anything/)]

## Usage
python 3.10 & pytorch 2.6.0 & CUDA version 12.4 verified

### 1. Installation

```bash
git clone https://github.com/robrosinc/REALTIME_SAM2.git
cd REALTIME_SAM2
conda create -n rtsam2 python=3.10
conda activate rtsam2
pip install -e .
```
### 2. Download Checkpoints

```bash
cd checkpoints
./download_checkpoints.sh
```

or EfficientTAM checkpoints are available at the [Hugging Face Space](https://huggingface.co/yunyangx/efficient-track-anything/tree/main).

### 3. Run tam_app.py inside 'notebooks' folder

⚠️ We recommend not using the reset button for now, instead try clicking on the object you want to change to multiple times.

## Performance
On a single 4070ti for inference,

sam2.1_hiera_tiny.pt takes 0.1 seconds for prompted frames / 0.08 seconds for non prompted frames.

efficienttam_ti_512x512.pt takes 0.025 seconds for prompted frames / 0.02 seconds for non prompted frames.

While tam is faster, using sam outputs masks with better quality.


## License
Efficient track anything checkpoints and codebase are licensed under [Apache 2.0](./LICENSE).
RobrosInc follows the requirements of Apache 2.0.

## Acknowledgement
Thank you to all the developers at Meta and github for contributing such an exciting project open source. 

+ [SAM2](https://github.com/facebookresearch/sam2)
+ [EfficientTAM](https://github.com/yformer/EfficientTAM)
+ [segment-anything-2-real-time](https://github.com/Gy920/segment-anything-2-real-time)

If you're using this repo in your research or applications, please cite EfficientTAM using this BibTeX:
```bibtex


@article{xiong2024efficienttam,
  title={Efficient Track Anything},
  author={Yunyang Xiong, Chong Zhou, Xiaoyu Xiang, Lemeng Wu, Chenchen Zhu, Zechun Liu, Saksham Suri, Balakrishnan Varadarajan, Ramya Akula, Forrest Iandola, Raghuraman Krishnamoorthi, Bilge Soran, Vikas Chandra},
  journal={preprint arXiv:2411.18933},
  year={2024}
}
```

## Dev Notes
We have enabled real-time addition of multiple objects.

April 07 2025 / reset button has been added, code cleanup, comments translated

April 17 2025 / preapring for modification (tracking with output_dict_per_obj to improve memory)
