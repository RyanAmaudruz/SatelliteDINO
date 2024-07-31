# SatelliteDINO

This repository is part of the Thesis work: Sky’s the Limit: Satellite Imagery Analysis with Image-level and Dense Self-Supervised Techniques.

## Abstract
The introduction of the Vision Transformer (ViT) has revolutionized the field of computer
vision, significantly advancing research in self-supervised learning (SSL). While SSL devel-
opments have predominantly focused on object-centric and RGB images, the application of
these methods to satellite imagery poses unique challenges due to substantial domain shifts.
This study explores the use of a plain ViT backbone for satellite image analysis as it presents
multiple advantages over its hierarchical version.
We investigated three SSL frameworks — DINO, Leopart, and ODIN — evaluating their
performance on satellite images. Our findings indicate that pretraining on satellite images
provides a substantial advantage over object-centric RGB images, underscoring the value
of domain-specific pretraining. We observed that advanced dense SSL algorithms did not
consistently outperform traditional image-level SSL frameworks, with fine-tuning results
highlighting limitations in the dense approach when adapted to a ViT backbone. Furthermore,
linear probing performance did not reliably predict fine-tuning outcomes, suggesting that
linear probing may not fully reflect real-world application performance.
Notably, the plain ViT backbone, when combined with our selected SSL frameworks, learned
powerful representations that outperformed recent benchmarks on the DFC2020 and MADOS
datasets. Future research could enhance this framework by integrating a ViT-Adapter with the
ODIN algorithm to improve object detection granularity and training efficiency. This approach
could also enable the ViT backbone to process multiple data modalities, offering promising
potential for further advancements in SSL. Additionally, integrating a Mask2Former decoder
with the ViT-Backbone for semantic segmentation could further improve performance in
instance, panoptic, and semantic segmentation, making the model more general and robust.

Full paper available upon request.

Author: *Amaudruz R.*

Supervisors: *Yuki A., Russwurm, M.*

## Acknowledgements
This project builds upon the work in the repository [SSL4EO-S12](https://github.com/zhu-xlab/SSL4EO-S12). The original research was conducted by Yi Wang, Nassim Ait Ali Braham, Zhitong Xiong, Chenying Liu, Conrad M. Albrecht, and Xiao Xiang Zhu, as part of their study titled "[SSL4EO-S12: A Large-Scale Multi-Modal, Multi-Temporal Dataset for Self-Supervised Learning in Earth Observation](https://ieeexplore.ieee.org/abstract/document/10261879)" (2023). We extend our gratitude to the authors for their contributions to the open-source community, which have provided a valuable foundation for this thesis.

## Access the pretraining dataset
- [x] **Raw dataset**: The full SSL4EO-S12 dataset (1.5TB, 500GB for each modality) is accessible at [mediaTUM](https://mediatum.ub.tum.de/1660427). There are some void IDs (gaps in folder names), see `data/void_ids.csv`. Center coordinates of all locations are available [here](https://drive.google.com/file/d/1RyJnGznSbMparS88BhHkXxETf0K-qYqI/view?usp=sharing).
- [x] **Example subset**: An example 100-patch subset (600MB) is available at [Google Drive](https://drive.google.com/file/d/1sRWcYbaWs-efXza6kw03GlJQdZHq5iRN/view?usp=sharing).
- [x] **Compressed dataset**: A compressed 8-bit version (20-50GB for each modality, including an RGB version) is available at [mediaTUM](https://mediatum.ub.tum.de/1702379). The raw 16/32-bit values are normalized by mean and std and converted to uint8. *Note: in our experiments, 8-bit input performs comparably well as 16-bit.*

## Visualising the pretraining dataset
Below, we show a few samples of the SSL4EO-S12 dataset under different band configuration. Information about the band configurations can be found [here](https://gisgeography.com/sentinel-2-bands-combinations/).

![Alt Text](visuals/ssl4eo_samples.png)

## Installation
We add conda yaml files to set up the Python environment under the [installation directory](https://github.com/your-username/your-repo-name/tree/main/installation).

## Contributions
- [x] **Refactoring**: Adapted the data loading for our purpose, added comments and performed cleaning.
- [x] **Data augmentations**: Updated the data augmentation. In particular, we create 2 special functions to randomly vary the saturation and contrast of the Multi-Spectral images used during the SSL pretraining phase.
- [x] **Visualisations**: Under the visualisations folder, we add some scripts that allowed the viewing of dataset samples and predictions.

## License
This repository is released under the Apache 2.0 license. The dataset and pretrained model weights are released under the CC-BY-4.0 license.
