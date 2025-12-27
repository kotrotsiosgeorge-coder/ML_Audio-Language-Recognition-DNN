# Spoken Language Identification – Deep Learning & MFCC

This project builds a **spoken language identification (LID) system** using **Deep Neural Networks (DNN)** and **Mel-Frequency Cepstral Coefficients (MFCC)**. It analyzes how linguistic similarity affects classification accuracy by comparing Romance languages with linguistically distant languages such as German and Japanese. :contentReference[oaicite:0]{index=0}

---

## 🎯 Goal
To investigate how **similarity between spoken languages** influences machine learning performance in language classification.

**Research Question:**  
➡️ *Does model accuracy decrease when distinguishing between similar languages compared to distant ones?*

---

## 📚 Dataset
- Source: **Mozilla Common Voice**
- ~4.7GB multilingual speech clips  
- Balanced sampling: **15,000 clips per language**
- Clip duration: ~3.5s → 10s
- Includes:
  - Romance languages: Spanish, Portuguese, Italian, Catalan, Romanian, French
  - Control languages: German, Japanese

---

## 🧹 Preprocessing
- Extract `.tar.gz` archives
- Clean and filter audio duration
- Normalize clip lengths
- Convert speech to **MFCC features**
- Build labeled feature matrix
- Equal class representation to avoid bias

MFCC computed with:
- **librosa**
- Sliding 10ms windows
- Spectrogram visualization
- Feature extraction pipelines

---

## 🧠 Model
**Dense Neural Network (DNN)** built with **TensorFlow / Keras**

### Architecture
- Fully connected Dense layers
- ReLU activations
- Dropout regularization
- Softmax output
- Adam optimizer
- Categorical cross-entropy

Designed to handle **high-dimensional MFCC inputs** while preventing overfitting.

---

## 📊 Experiments & Results

| Experiment | Languages | Accuracy | Notes |
|----------|----------|---------|------|
| Model 1 | Spanish vs Portuguese | **98%** | Similar Romance pair |
| Model 2 | + Italian + French | **84%** | Drops due to confusion |
| Model 3 | + Catalan + Romanian | **73%** | Multi-class complexity |
| Model 4 | Romanian + German + Japanese | **90%** | Distant languages easier |
| Model 5 | Spanish + German + Japanese | **94%** | Romanian hardest |
| Model 6 | Portuguese + Spanish + Italian | **98%** | Romance languages perform best |

### Key Findings
✔️ Similar Romance languages surprisingly classify **better** than expected  
✔️ Romanian consistently harder to classify  
✔️ Linguistic similarity ≠ phonetic similarity  
✔️ MFCC features capture **acoustic**, not linguistic relationships  

---

## 🧠 Interpretation
- MFCC encodes phonetics, not grammar
- Romance languages share stable acoustic patterns
- Romanian has larger phonetic variability
- Class complexity increases misclassification risk
- Dataset balance & clip quality matter

---

## ⚠️ Ethical Considerations
- Bias in crowdsourced voice data
- Potential discrimination risk if misused
- Need fair + inclusive model evaluation

---

## 🛠️ Tech Stack
- Python
- TensorFlow / Keras
- librosa
- NumPy
- Pandas
- Matplotlib

---

## 📄 Report
Full academic report with methodology, analysis, and evaluation included:
- `ML_Exam_Project.pdf`

---

## 👨‍💻 Authors
CBS MSc Data Science — Machine Learning Exam Project  
Team including **Georgios Kotrotsios**

---

## ⭐ Takeaway
A deep learning–based LID system can achieve **up to 98% accuracy**, but performance depends heavily on language grouping and phonetic characteristics—not just linguistic similarity.

