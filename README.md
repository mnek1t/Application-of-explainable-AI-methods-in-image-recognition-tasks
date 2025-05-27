# APPLICATION OF EXPLAINABLE AI METHODS IN AN IMAGE RECOGNITION TASK

## This is the source code developed for the Bachelor's Thesis practical part implementation.

---

### Author  
**Mykyta Medvediev**, student of Riga Technical University

### Scientific Advisor  
**Dr.sc.ing., Dr.paed., Professor Alla Anohina-Naumeca**

---

## XAI Methods Implemented

- SHAP  
- LIME  
- Permutation Importance  
- Grad-CAM  
- Trainable Attention  
- LRP (Alpha-Beta Variant)  
- DeepLIFT  

---

## Getting Started

### Clone the Repository

```bash
git clone https://github.com/mnek1t/Application-of-explainable-AI-methods-in-image-recognition-tasks
cd Application-of-explainable-AI-methods-in-image-recognition-tasks
```

### Install all required libraries
```bash
pip install tensorflow, keras, shap, lime, sklearn, quantus, matplotlib, cv2, numpy
```

### For LRP alpha-beta
Install Python 3.9 version
Download Python 3.9 from the official site:<br/>
https://www.python.org/downloads/release/python-390/
Install innvestigate library 
```bash
pip install innvestigate
```

Create a virtual envirnoment to avoid version conflict:<br/>
However be sure you installed Python version from https://www.python.org/downloads/release/python-390/
```bash
python3.9 -m venv xai_env
source xai_env/bin/activate  # On Windows use: xai_env\Scripts\activate
```

Run the file
```bash
python lrp_implementation.py
```

Make sure you use correct paths locations for constants defined in utils folder

## Notes
The model was trained using an augmented garbage image dataset of 4200 entries.<br/>
All methods were applied to a custom Inception v3-based architecture using transfer learning.<br/>
Evaluation metrics and explainability assessments were conducted using the Quantus library.<br/>
