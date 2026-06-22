# 🔍 Perineal Tear Detection (PELVITRACK)

## 📌 Overview
This project aims to detect perineal tissue rupture from video sequences using computer vision and machine learning.

Developed within the European research project **PELVITRACK**.

---

## 🎯 Objectives
- Detect rupture as early as possible in video sequences
- Extract meaningful features from images
- Build a robust predictive pipeline

---

## 🧠 Methods

### 📊 Data
- 13 experimental videos of perineal tissue (pig model)
- Cyclic stretching until rupture

### ⚙️ Pipeline
1. Frame extraction (temporal sampling)
2. Feature extraction:
   - Intensity (mean, std)
   - Texture (Laplacian)
   - Motion (optical flow)
   - Temporal features (cycle position)
3. Model training:
   - Gradient Boosting
4. Evaluation:
   - Leave-One-Video-Out validation

---

## 📈 Results
- **Accuracy:** 91.3%
- **F1-score:** 91.5%
- Detection delay: ~1.5s

---

## 🧪 Features
Main important feature:
- `img1_rise`: increase of brightness from local minimum

---

## 🚀 Usage

```bash
pip install -r requirements.txt
python src/main.py
