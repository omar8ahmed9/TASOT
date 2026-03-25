# TASOT: Multimodal Optimal Transport for Unsupervised Temporal Segmentation in Surgical Robotics

Official implementation of **TASOT**, a multimodal extension of ASOT for **unsupervised temporal segmentation** in surgical videos.

For method details, experiments, and full results, please refer to the paper:

**Multimodal Optimal Transport for Unsupervised Temporal Segmentation in Surgical Robotics**  
Omar Mohamed, Edoardo Fazzari, Ayah Al-Naji, Hamdan Alhadhrami, Khalfan Hableel, Saif Alkindi, Cesare Stefanini  
**arXiv preprint arXiv:2602.24138, 2026**

---

## Overview

Temporal segmentation in surgical videos aims to partition long procedures into meaningful phases or steps. Recent approaches often rely on large-scale pretraining and zero-shot transfer, which can be effective but also computationally expensive.

TASOT addresses this by incorporating language generated directly from the videos as an additional source of supervision, without requiring manual annotations or heavy surgical pretraining. By combining visual and textual cues in a multimodal optimal transport framework, TASOT achieves strong performance and outperforms existing zero-shot methods on multiple surgical benchmarks.

The overall TASOT pipeline is illustrated below:

<p align="center">
  <img src="tasot_overview.png" alt="TASOT overview" width="900"/>
</p>

**Figure:** Overview of TASOT. Surgical videos are divided into temporal windows, captions are generated for each clip, visual features are extracted with DINOv3, textual features are extracted with CLIP, and both modalities are fused in the optimal transport objective for unsupervised temporal segmentation.



## Installation

Create the environment and install dependencies:

```bash
conda create -n tasot python=3.10 -y
conda activate tasot
pip install -r requirements.txt
```


## Pipeline

### 1. Captioning pipeline

Run the scripts in this order:

```text
scripts/captioning_pipeline/
1. generate_window_plans.py
2. cut_video_windows.py
3. generate_gemini_captions.py
4. merge_window_captions.py
```
### 2. Feature extraction

```text
scripts/feature_extraction/
- embed_clip_captions.py
- extract_dinov3_features.py
```

### 3. Training

```text
scripts/run_scripts/
- run_cholec80_dinov3_clip.sh
- run_autolapro_dinov3_clip.sh
- run_mb140_dinov3_clip.sh
```



## Datasets

This repository supports:
- Cholec80
- AutoLaparo
- MultiBypass140

Please prepare the datasets according to their official access policies and place files in the expected folder structure.



## Citation

If you use this repository in your research, please cite:

```bibtex
@article{mohamed2026multimodal,
  title={Multimodal Optimal Transport for Unsupervised Temporal Segmentation in Surgical Robotics},
  author={Mohamed, Omar and Fazzari, Edoardo and Al-Naji, Ayah and Alhadhrami, Hamdan and Hableel, Khalfan and Alkindi, Saif and Stefanini, Cesare},
  journal={arXiv preprint arXiv:2602.24138},
  year={2026}
}
```
