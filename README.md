# TUKL_2025

Repository to track my progress throughout internship at TUKL-DLL, @NCAI, NUST

## Active Learning

I implemented active learning for crop classification using satellite time-series imagery. A CNN-based model was trained to classify crop types using only a **small fraction of labeled data**, reducing annotation cost.

📌 **Key Highlights**:
- Dataset: 20,000 satellite time-series samples
- Task: Multiclass crop classification (3 classes)
- Model: 1D CNN + Fully Connected classifier
- Input: Temporal sensor data (e.g., NDVI over time)
- Strategy: **Margin Sampling** (uncertainty-based)

🎯 **Results**:
Using only **2.5% of the data** (500 labeled samples), the model achieved:

> ✅ **Accuracy: 86.4%**

This demonstrates the power of active learning to minimize labeling effort while maintaining high accuracy.

---

### 📈 Performance Graph

![Accuracy vs Labeled Samples](assets/accuracy_vs_labeled_samples.png)
