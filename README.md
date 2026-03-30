# BAMBI_MOT
# BAMBI Wildlife MOT

Benchmarking and improving Multi-Object Tracking for aerial wildlife 
monitoring using the BAMBI dataset.

**Course:** CIV4  
**Group:** Stanislav Buinitski, Yahaya Danjuma, Navid Ghaderian, Afzaal Yasin
**Dataset:** https://www.bambi.eco/

---

## Project Overview

We evaluate state-of-the-art MOT trackers (SORT, ByteTrack, BoT-SORT) 
on nadir UAV wildlife footage and analyze domain-specific failure modes. 
We then implement a targeted improvement based on our findings.

---

## Setup

### 1. Clone the repo
git clone https://github.com/Navid-alt/BAMBI_MOT.git

# Local development
pip install -e ".[dev]"

# On Google Colab
pip install -e ".[colab]"


### Data

Data is not tracked in this repository. Download instructions are in
`notebooks/01_data_pipeline.ipynb`. Configure paths in `config.yaml` before
running anything.

---

## Pipeline

| Step | Notebook | Description |
|---|---|---|
| 1. Data | `01_data_pipeline.ipynb` | Download, extract frames, prepare annotations |
| 2. Detection | `02_detection.ipynb` | Train and evaluate YOLOv8 on thermal frames |
| 3. Tracking | `03_tracking.ipynb` | Run SORT, ByteTrack, BoT-SORT |
| 4. Evaluation | `04_evaluation.ipynb` | HOTA/MOTA/IDF1 metrics and failure analysis |
---

## Results

*(filled in as project progresses)*

---

## References
- BAMBI Dataset: https://github.com/bambi-eco/Dataset
- ByteTrack: Zhang et al., ECCV 2022
- BoT-SORT: Aharon et al., arXiv 2022
- HOTA: Luiten et al., IJCV 2020
---
