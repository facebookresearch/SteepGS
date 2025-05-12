# Steepest 3D Gaussian Splatting for Compact Radiance Field

The official implementation of CVPR 2025 paper [Steepest Descent Density Control for Compact 3D Gaussian Splatting
](https://arxiv.org/abs/2505.05587)

[Peihao Wang*](https://peihaowang.github.io/)<sup>1</sup>,
[Yuehao Wang*](https://yuehaolab.com/)<sup>1</sup>,
[Dilin Wang](https://wdilin.github.io/)<sup>2</sup>,
[Sreyas Mohan](https://sreyas-mohan.github.io/)<sup>2</sup>,
[Zhiwen Fan](https://zhiwenfan.github.io/)<sup>1</sup>,
[Lemeng Wu](https://sites.google.com/view/lemeng-wu/home)<sup>2</sup>,
[Ruisi Cai](https://cairuisi.github.io/)<sup>1</sup>,
[Yu-Ying Yeh](https://yuyingyeh.github.io/)<sup>2</sup>,
[Zhangyang Wang](https://vita-group.github.io/)<sup>1</sup>,
[Qiang Liu](https://www.cs.utexas.edu/~lqiang/)<sup>1</sup>,
[Rakesh Ranjan](https://scholar.google.com/citations?user=8KF99lYAAAAJ&hl=en)<sup>2</sup>

<sup>1</sup>University of Texas at Austin, <sup>2</sup>Meta Reality Labs

<sup>*</sup> denotes equal contribution.

[Project Page](https://vita-group.github.io/SteepGS/) | [Paper](https://arxiv.org/abs/2505.05587) | [Code](https://github.com/facebookresearch/SteepGS)

This repository is built based on the [official repository of 3DGS](https://github.com/graphdeco-inria/gaussian-splatting).

## Get Started

### Cloning the Repository

The first step is to clone this repository by:
```shell
git clone https://github.com/facebookresearch/SteepGS
```

Unlike the original repository, the `diff-gaussian-rasterization` and `simple-knn` libraries are already in the repo.

```shell
submodules/diff-gaussian-rasterization
submodules/simple-knn
```

However, please remember to manually clone `glm` library via:
```shell
cd submodules/diff-gaussian-rasterization/third_party
git clone https://github.com/g-truc/glm.git
git checkout 5c46b9c
```


### Installation

Running our code requires the following packages:

```shell
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
conda install nvidia/label/cuda-11.8.0::cuda # optional, for nvcc toolkits
```

You also need to install two customized packages `diff-gaussian-rasterization` and `simple-knn`:

```shell
# remember to specify the cuda library path if some cuda header is missing
cd submodules/diff-gaussian-rasterization
pip install -e .

# remember to specify the cuda library path if some cuda header is missing
cd submodules/simple-knn
pip install -e .
```

### Data Preparation

The data downloading and processing are the same with the original 3DGS. Please refer to [here](https://github.com/graphdeco-inria/gaussian-splatting?tab=readme-ov-file#running) for more details. If you want to run SteepGS on your own dataset, please refer to [here](https://github.com/graphdeco-inria/gaussian-splatting?tab=readme-ov-file#processing-your-own-scenes) for the instructions.


## Running

The simplest way to use and evaluate SteepGS is through the following commands:

```shell
python train.py -s <path to COLMAP or NeRF Synthetic dataset> -m <path to checkpoint> --no_gui --density_strategy steepest  --eval # Train with train/test split
python render.py -m <path to trained model> # Generate renderings
python metrics.py -m <path to trained model> # Compute error metrics on renderings
```

SteepGS inherits all training hyper-parameters from original 3DGS, listed [here](https://github.com/graphdeco-inria/gaussian-splatting?tab=readme-ov-file#running) in details. In addition, SteepGS introduces a few arguments associated with the steepest density control strategy:

- `--densify_strategy`: The strategy adopted for density control. It can be `adc` to recover the default density control in 3DGS or `steepest` to enable our method. Users can also append attributes `stationary` to enable stationary gradient condition, `no_saddle`, `no_uncertain`, `no_eig_cond` to disable splitting conditions on saddle points, gradient uncertainty, or splitting matrices' eigenvalues, or `no_eig_upd` to disable adopting splitting matrices' principal eigenvectors as the update directions.
- `--densify_S_threshold`: The threshold of splitting matrice's eigenvalues used to select Gaussian points to be split. It must be negative.
- `--S_estimator`: The splitting matrix estimator. It can be `partial`, `approx`, or `inv_cov`. By default, `inv_cov` is chosen for its computational efficiency.


## Citation

If you find our repository helpful, please cite our work using the following BibTex.

```
@inproceedings{wang2025steepgs,
  title={Steepest Descent Density Control for Compact 3D Gaussian Splatting},
  author={Wang, Peihao and Wang, Yuehao and Wang, Dilin and Mohan, Sreyas and Fan, Zhiwen and Wu, Lemeng and Cai, Ruisi and Yeh, Yu-Ying and Wang, Zhangyang and Liu, Qiang and Ranjan, Rakesh},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025}
}
```
