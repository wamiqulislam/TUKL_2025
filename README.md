# TUKL_2025

Repository to track my progress throughout internship at TUKL-DLL, @NCAI, NUST

## 🔁 Active Learning

I implemented **Active Learning** for **satellite image time series classification** to identify crop types. The key idea was to **train a CNN-based classifier** using only a *small, informative subset* of the available data instead of labeling everything, which is slow and expensive to do.

### ✅ What I Did:

- Applied **multiple active learning strategies**:
  - **Uncertainty-based** (e.g., *margin sampling*, *entropy*, *least confidence*)
  - **Diversity-based**
  - **Density-based**
  - **Hybrid strategies**
- Used a **1D CNN model** trained on temporal pixel data
- Integrated a framework to iteratively query the most informative samples
- Achieved **86.4% accuracy using just 2.5%** (500 out of 20,000) of the data with *margin sampling*

### 📉 Results:

Even with limited labeled data, the model rapidly improved as shown in the graph below:

![Active Learning Accuracy](https://github.com/wamiqulislam/TUKL_2025/blob/main/Active%20Learning/Plots/300%2C20%2C10%2Ccomplete.png)

> These plots show accuracy and F1 score vs. number of labeled samples for different strategies.

---

This approach significantly reduces labeling effort while maintaining high performance—especially useful in resource-constrained remote sensing applications.
