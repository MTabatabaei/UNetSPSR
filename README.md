
# UNetSPSR

---

Python implementation of U-Net Structure-Preserving Super Resolution, trained on craniofacial CBCT imaging data


---

## Citation

If you use this code in your work, please cite:

> S. Hosseinitabatabaei, A.J. Nelson, N. Piché, D. Dagdeviren, N. Reznikov,  
> [*Craniofacial CBCT: Addressing Volume-Resolution Dilemma using Generative AI.*](tbd)
> **Journal of Dental Research (JDR), 2025**

<br/><br/>
**UNetSPSR** is an adaptation of SPSR ([Ma et. al., 2020](https://github.com/Maclory/SPSR/tree/master)) with key differences.  
Similar to SPSR, which was trained on natural images, a dedicated gradient branch helped the network learn edge information, enhancing the delineation of fine structures like cortical bone boundaries and trabecular architecture. The main difference is the replacement of the standard discriminator with the U-Net-style discriminator from Real-ESRGAN ([Wang et. al., 2021](https://github.com/xinntao/Real-ESRGAN)) improved the network’s ability to penalize artifacts and suppress false textures, crucial for clinical reliability. These combined strategies enhanced both structural fidelity and perceptual quality while reducing typical GAN issues such as hallucinated structures and training instability.

---

## Dependencies


- Python 3 (Anaconda recommended)

- PyTorch >= 1.0
- NVIDIA GPU + CUDA
- Python packages: pip install numpy opencv-python lmdb pyyaml SimpleITK itk tifffile
- TensorBoard

---

## Dataset Preparation

### Preprocess Datasets:

### For training or testing with performance evaluation *(low-res and high-res images required)*:
0. If simulate LR images from the HR images, skip steps 1 and 2. The package will automatically generate LR images. If more sophisticated degraded images are needed, you need to generate them separately, and go to step 3.
1. When both the LR and HR CBCT images are required, it is important to meticulously register them.
We used  [Dragonfly 3D-World software](https://dragonfly.comet.tech/) (Comet Technologies Canada Inc.).
2. The aligned LR and HR images need to be cropped to their common region. 
3. To address potential edge artifacts and large gradient values that could disrupt the training of the GAN, the blank regions 
(i.e., outside the HR FOV) need to be padded using an inpainting 
algorithm (OpenCV library).
4. Finally, image intensities need to be normalized to a [0, 1] range.
> 💡The script to perform steps 3 and 4 can be found [here](tbd)


### For testing *(only low-res images required)*:
1. For best results, the LR images need to be smoothed mildly using Gaussian smoothing, with a small sigma (e.g., 0.5)
2. To address potential edge artifacts and large gradient values that could disrupt the training of the GAN, the blank regions 
(i.e., outside the HR FOV) need to be padded using an inpainting algorithm (OpenCV library).
3. Image intensities need to be normalized to a [0, 1] range.
> 💡The script to perform steps 1-3 can be found [here](tbd)

> 💡UNetSPSR has been training using data augmented not only by random flips and rotations,
> but also random changes in brightness, contrast, and gamma, making it robust to changes in image
> contrast and raw values. However, as with any deep learning model, the best testing performance
> is achieved by ensuring that the images have similar ranges and contrasts to those used for training.
> 
> For *J. Morita 3D Accuitomo-170* scanner, we used the following normalization formula to maximize the contrast in images to facilitate training:
    $normalized\;value = (raw\;value-26000)/(46000-26000)$
> 
> In our case for the CBCT scans acquired using a J. Morita 3D Accuitomo-170 scanner (Morita Co, Kyoto, Japan)
> we empirically based on an empirically identified intensity range for normalization to be 26,000 to 46,000, selected to 
enhance contrast and ensure inclusion of relevant anatomical structures.
> 
> Below is an example of the normalized
> values for different tissues in our scans before and after normalization between *26,000* and *46,000*:
> 
> 
> 
> **If using scans from another scanner manusfacturer for training, these values may need to be determined, and modified in the json file.
> If values are not known, normalization using 0 and 65,000 works too, although training may become harder.**

---

## Training
To train an UNetSPSR model, run this command:
   ```bash
   python train.py -opt options/train/train_spsr.json
   ```

- The json file will be processed by options/options.py. Details on the json file are provided below.

- Before running this code, please modify train_spsr.json to your own configurations (details in following table).

- You can find your training results in ./experiments.

- During training, you can use Tesorboard to monitor the losses with tensorboard --logdir tb_logger/NAME_OF_YOUR_EXPERIMENT


### The input train_unetspsr.json file
```json
 {
  "name": "UNetSPSR"
  , "use_tb_logger": true
  , "model":"unetspsr"
  , "scale": 2
  , "gpu_ids": [0]

  , "datasets": {
    "train": {
      "name": "TRAIN"
      , "mode": "LRHR"
      , "dataroot_HR": "\\\\path\\to\\training\\HR\\image\\slices"
      , "dataroot_LR": "\\\\path\\to\\training\\LR\\image\\slices"
      , "subset_file": null
      , "use_shuffle": true
      , "n_workers": 8
      , "batch_size": 32
      , "HR_size": 128
      , "use_flip": true
      , "use_rot": true
      , "use_contrast": true
      , "normalization_range": [26000,46000]
    }
    , "val": {
      "name": "VAL"
      , "mode": "LRHR"
      , "dataroot_HR": "\\\\path\\to\\validation\\HR\\image\\slices"
      , "dataroot_LR": "\\\\path\\to\\validation\\LR\\image\\slices"
    }
  }

  , "path": {
    "root": "\\\\root\\path\\UNetSPSR_model" //modify to you own root path
    // , "resume_state": "../experiments/UNetSPSR/training_state/25000.state"
    , "pretrain_model_G": null
  }

  , "network_G": {
    "which_model_G": "unetspsr_net"
    , "norm_type": null
    , "mode": "CNA"
    , "nf": 64
    , "nb": 23
    , "in_nc": 1
    , "out_nc": 1
    , "gc": 32
  }

  , "network_D": {
    "which_model_D": "discriminator_unet_128"
    , "norm_type": "batch"
    , "act_type": "leakyrelu"
    , "mode": "CNA"
    , "nf": 64
    , "in_nc": 1
  }

  , "train": {
    "lr_G": 1e-4
    , "lr_G_grad": 1e-4
    , "weight_decay_G": 0
    , "weight_decay_G_grad": 0
    , "beta1_G": 0.9
    , "beta1_G_grad": 0.9
    , "lr_D": 1e-4
    , "weight_decay_D": 0
    , "beta1_D": 0.9
    , "lr_scheme": "MultiStepLR"
    , "lr_steps": [50000, 100000, 200000, 300000]
    , "lr_gamma": 0.5

    , "pixel_criterion": "l1"
    , "pixel_weight": 0.3
    , "feature_criterion": "l1"
    , "feature_weight": 1
    , "gan_type": "lsgan"
    , "gan_weight": 5e-3
    , "gradient_pixel_weight": 1e-2
    , "gradient_gan_weight": 5e-3
    , "pixel_branch_criterion": "l1"
    , "pixel_branch_weight": 5e-1
    , "Branch_pretrain" : 0
    , "Branch_init_iters" : 5000

    , "manual_seed": 6
    , "niter": 5e5
    , "val_freq": 5e3
  }

  , "logger": {
    "print_freq": 100
    , "save_checkpoint_freq": 5e3
  }
}
```

### 📝 Parameter Descriptions

| Key | Description                                                                                                     |
|-----|-----------------------------------------------------------------------------------------------------------------|
| `name` | Name of the experiment or model configuration (e.g., `UNetSPSR`)                                                |
| `use_tb_logger` | Whether to use TensorBoard logging (recommend `true`, or `false`)                                               |
| `model` | Name of the model type (e.g., `unetspsr`)                                                                       |
| `scale` | Integer upscaling factor for super-resolution (e.g., `2`)                                                       |
| `gpu_ids` | List of GPU IDs to use (e.g., `[0]` for 1 GPU, or `[3,2,1,0]` for 4 GPUs)                                       |
| `datasets.train.name` | Identifier for training dataset                                                                                 |
| `datasets.train.mode` | Mode of dataset input, `LRHR` for real low- and high-resolution pairs, `HR` for simulated low-resolution images |
| `datasets.train.dataroot_HR` | Path to high-resolution training image slices                                                                   |
| `datasets.train.dataroot_LR` | Path to low-resolution training image slices (if `LRHR` mode)                                                   |
| `datasets.train.subset_file` | File path for training subset list (or `null`)                                                                  |
| `datasets.train.use_shuffle` | Whether to shuffle training data (recommend `true`, or `false`)                                                 |
| `datasets.train.n_workers` | Number of worker threads for data loading (e.g., `8`)                                                           |
| `datasets.train.batch_size` | Batch size used during training (e.g., `32`)                                                                    |
| `datasets.train.HR_size` | The size of HR training patches for training (e.g., `128`)                                                      |
| `datasets.train.use_flip` | Whether to apply random horizontal/vertical flips (recommend `true`, or `false`)                                |
| `datasets.train.use_rot` | Whether to apply random 90° rotations (recommend `true`, or `false`)                                            |
| `datasets.train.use_contrast` | Whether to apply random contrast augmentation (recommend `true`, or `false`)                                    |
| `datasets.train.normalization_range` | Intensity normalization range (e.g., `[26000, 46000]`)                                                          |
| `datasets.val.name` | Identifier for validation dataset                                                                               |
| `datasets.val.mode` | Mode of dataset input for validation (`LRHR` or `HR`)                                                           |
| `datasets.val.dataroot_HR` | Path to high-resolution validation slices                                                                       |
| `datasets.val.dataroot_LR` | Path to low-resolution validation slices                                                                        |
| `path.root` | Root directory to save experiment artifacts                                                                     |
| `path.pretrain_model_G` | Path to pretrained generator model (or `null`)                                                                  |
| `network_G.which_model_G` | Name of generator architecture (e.g., `unetspsr_net`)                                                           |
| `network_G.norm_type` | Normalization type (if used, else `null`)                                                                       |
| `network_G.mode` | Convolution mode (recomment `CNA`[Convolution -> Normalization -> Activation], etc.)                            |
| `network_G.nf` | Number of base filters or features (e.g., `64`)                                                                 |
| `network_G.nb` | Number of residual blocks (e.g., `23`)                                                                          |
| `network_G.in_nc` | Number of input image channels (for grayscale: `1`, for color: `3`)                                             |
| `network_G.out_nc` | Number of output image channels (for grayscale: `1`, for color: `3`)                                            |
| `network_G.gc` | RRDB block growth channel (i.e., intermediate channels) (e.g., `32`)                                            |
| `network_D.which_model_D` | Name of discriminator architecture (e.g., `discriminator_unet_128`)                                             |
| `network_D.norm_type` | Normalization type used in discriminator (`batch`, etc.)                                                        |
| `network_D.act_type` | Activation type (`leakyrelu`, etc.)                                                                             |
| `network_D.mode` | Convolution mode in discriminator (`CNA`, etc.)                                                                 |
| `network_D.nf` | Number of base filters (e.g., `64`)                                                                             |
| `network_D.in_nc` | Number of input channels for discriminator (for grayscale: `1`, for color: `3`)                                 |
| `train.lr_G` | Learning rate for generator (e.g., `1e-4`)                                                                      |
| `train.lr_G_grad` | Learning rate for gradient branch (e.g., `1e-4`)                                                                |
| `train.weight_decay_G` | Weight decay for generator (e.g., `0`)                                                                          |
| `train.weight_decay_G_grad` | Weight decay for gradient branch (e.g., `0`)                                                                    |
| `train.beta1_G` | Beta1 for Adam optimizer (generator) (e.g., `0.9`)                                                              |
| `train.beta1_G_grad` | Beta1 for Adam optimizer (gradient branch) (e.g., `0.9`)                                                        |
| `train.lr_D` | Learning rate for discriminator (e.g., `1e-4`)                                                                  |
| `train.weight_decay_D` | Weight decay for discriminator (e.g., `0`)                                                                      |
| `train.beta1_D` | Beta1 for Adam optimizer (discriminator) (e.g., `0.9`)                                                          |
| `train.lr_scheme` | Learning rate schedule type (e.g., `MultiStepLR`)                                                               |
| `train.lr_steps` | List of iteration steps to drop LR (e.g., `[50000, 100000, 200000, 300000]`)                                    |
| `train.lr_gamma` | Multiplicative factor for LR decay (e.g., `0.5`)                                                                |
| `train.pixel_criterion` | Loss function for pixel-wise loss (e.g., `l1`)                                                                  |
| `train.pixel_weight` | Weight of pixel-wise loss (e.g., `0.3`)                                                                         |
| `train.feature_criterion` | Loss function for feature loss (e.g., `l1`)                                                                     |
| `train.feature_weight` | Weight of feature loss (e.g., `1`)                                                                              |
| `train.gan_type` | GAN type used (e.g., `lsgan`)                                                                                   |
| `train.gan_weight` | Weight of GAN loss (e.g., `5e-3`)                                                                               |
| `train.gradient_pixel_weight` | Weight of gradient branch pixel loss (e.g., `1e-2`)                                                             |
| `train.gradient_gan_weight` | Weight of gradient branch GAN loss (e.g., `5e-3`)                                                               |
| `train.pixel_branch_criterion` | Loss function used in branch pretraining (e.g., `l1`)                                                           |
| `train.pixel_branch_weight` | Weight of pixel loss in the branch (e.g., `5e-1`)                                                               |
| `train.Branch_pretrain` | Whether to pretrain branch first (`0` = no)                                                                     |
| `train.Branch_init_iters` | Number of iterations to initialize the branch (e.g., `5000`)                                                    |
| `train.manual_seed` | Random seed for reproducibility (e.g., `9`)                                                                     |
| `train.niter` | Total number of training iterations (e.g., `5e5`)                                                               |
| `train.val_freq` | Frequency of validation (in iterations) (e.g., `5e3`)                                                           |
| `logger.print_freq` | Frequency of printing log messages (e.g., `100`)                                                                |
| `logger.save_checkpoint_freq` | Frequency of saving model checkpoints (e.g., `5e3`)                                                             |

---

## Testing
To train an UNetSPSR model, run this command:
   ```bash
   python train.py -opt options/train/train_spsr.json
   ```

- The json file will be processed by options/options.py. Details on the json file are provided below.

- Before running this code, please modify train_spsr.json to your own configurations (details in following table).

- You can find your training results in ./experiments.

- During training, you can use Tesorboard to monitor the losses with tensorboard --logdir tb_logger/NAME_OF_YOUR_EXPERIMENT


### The input test_unetspsr.json file
```json
 {
  "name": "UNetSPSR"
  , "model": "unetspsr"
  , "scale": 2
  , "gpu_ids": [0]

  , "datasets": {
    "test_1": { 
      "name": "test_name"
      , "mode": "LRHR"
      , "dataroot_HR": "\\\\path\\to\\training\\HR\\image\\slices"
      , "dataroot_LR": "\\\\path\\to\\training\\LR\\image\\slices"
    }
  }
  , "path": {
    "root": "\\\\root\\path\\UNetSPSR_model" //modify to you own root path
    // , "resume_state": "../experiments/UNetSPSR/training_state/25000.state"
    , "pretrain_model_G": null
  }

  , "network_G": {
    "which_model_G": "unetspsr_net"
    , "norm_type": null
    , "mode": "CNA"
    , "nf": 64
    , "nb": 23
    , "in_nc": 1
    , "out_nc": 1
    , "gc": 32
  }
}

```
| Key | Description                                                           |
|-----|-----------------------------------------------------------------------|
| `name` | Name of the experiment or model configuration (e.g., `UNetSPSR`)      |
| `model` | Name of the model type (e.g., `unetspsr`)                             |
| `scale` | Integer upscaling factor for super-resolution (e.g., `2`)             |
| `gpu_ids` | List of GPU IDs to use (e.g., `[0]`)                                  |
| `datasets.test_1.name` | Identifier for test dataset (e.g., `test_name`)                       |
| `datasets.test_1.mode` | Mode of dataset input, `LRHR` for real low- and high-resolution pairs |
| `datasets.test_1.dataroot_HR` | Path to high-resolution test image slices                             |
| `datasets.test_1.dataroot_LR` | Path to low-resolution test image slices                              |
| `path.root` | Root directory to save experiment artifacts                           |
| `path.pretrain_model_G` | Path to pretrained generator model                                    |
| `network_G.which_model_G` | Name of generator architecture (e.g., `unetspsr_net`)                 |
| `network_G.norm_type` | Normalization type (if used, else `null`)                             |
| `network_G.mode` | Convolution mode (e.g., `CNA`)                                        |
| `network_G.nf` | Number of base filters or features (e.g., `64`)                       |
| `network_G.nb` | Number of residual blocks (e.g., `23`)                                |
| `network_G.in_nc` | Number of input image channels (e.g., `1`)                            |
| `network_G.out_nc` | Number of output image channels (e.g., `1`)                           |
| `network_G.gc` | RRDB block growth channel (e.g., `32`)                                |

---
## Results

### UNetSPSR network outperforms other state-of-the-art methods


### UNetSPSR reduces the overestimation errors in mandibular and maxillary bone morphological measurements


### UNetSPSR partially recovers visualization of pulp chamber and root canals

---
## Acknowledgement 
This code is forked and modified from [SPSR](https://github.com/Maclory/SPSR/tree/master)

---
## Contact 
[For any questions, contact mahdi.tabatabaei@mail.mcgill.ca](mailto:mahdi.tabatabaei@mail.mcgill.ca)

