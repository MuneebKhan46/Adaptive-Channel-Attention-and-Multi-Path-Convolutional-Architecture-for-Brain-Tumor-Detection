
# **Adaptive Channel Attention and Multi-Path CNN Architecture for Brain Tumor Detection**

This project presents a **deep learning-based approach** to detect brain tumors from MRI images. The proposed model integrates **Efficient Channel Attention (ECA)** and **Convolutional Block Attention Module (CBAM)** within a multi-path convolutional architecture to achieve high classification accuracy. 

Leveraging the **Brain Tumor MRI Dataset** from Kaggle, the network classifies brain tumors into four categories: **glioma, meningioma, pituitary tumor, and no tumor**. 

---

## **Project Overview**

The goal of this project is to explore how **attention mechanisms** like ECA and CBAM can enhance feature extraction in CNNs, improving the performance of brain tumor classification tasks. The dataset consists of MRI scans organized into four distinct categories, and the trained model helps automate the identification of tumor types.

This repository contains:
- **Implementation of a Multi-Path CNN Model** with ECA and CBAM attention mechanisms.
- **Training, Evaluation, and Visualization Scripts** to analyze model performance.
- **Reproducible Results** using a pre-organized data pipeline and training configurations.

---

## **Repository Structure**

```
Adaptive-Channel-Attention-for-Brain-Tumor-Detection/
├── data_loader.py        # Handles data loading and preprocessing
├── model.py              # Defines the CNN with ECA and CBAM blocks
├── train.py              # Script to train the model
├── metrics.py            # Evaluation metrics (accuracy, precision, recall, etc.)
├── plots.py              # Visualization tools (confusion matrix, ROC curve)
├── requirements.txt      # List of required dependencies
└── README.md             # Project documentation
```

---

## **Getting Started**

### **1. Clone the Repository**

To clone the repository and navigate into the project directory, run:

```bash
git clone https://github.com/yourusername/Adaptive-Channel-Attention-for-Brain-Tumor-Detection.git
cd Adaptive-Channel-Attention-for-Brain-Tumor-Detection
```

---

### **2. Install Dependencies**

Ensure you have **Python 3.x** installed. Install all required dependencies with:

```bash
pip install -r requirements.txt
```

#### **Required Libraries:**
- **TensorFlow 2.12.0**
- **NumPy**
- **Pandas**
- **Matplotlib**
- **Seaborn**
- **Scikit-learn**
- **Pillow**  

---

### **3. Dataset Setup**

This project uses the **Brain Tumor MRI Dataset** available on Kaggle. Follow these steps to download and organize the dataset:

1. **Dataset Link**: [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
2. **Extract the dataset** into your project directory:
   
   ```
   brain-tumor-mri-dataset/
   ├── Training/
   │   ├── glioma/
   │   ├── meningioma/
   │   ├── pituitary/
   │   └── no_tumor/
   └── Testing/
   ```
---

### **4. Training and Testing**

Use the following commands to train, test, and visualize the model with the following commands:

1. **Train the Model** 

   ```bash
   python train.py
   ```

2. **Test the Model**  
   Evaluate the trained model on the test data and display performance metrics.

   ```bash
   python metrics.py
   ```

3. **Visualize Results**  
   Generates a **confusion matrix** and **ROC curves** to visualize the model’s performance.

   ```bash
   python plots.py
   ```

---

## **Results and Visualization**

- **Confusion Matrix**: Provides insights into the model's predictions across the four categories.  
- **ROC Curve**: Measures the trade-off between true positive and false positive rates for each class.  
- **Evaluation Metrics**: Includes **accuracy**, **precision**, **recall**, **F1-score**, and **mAP** (mean average precision).

---

## **Contributing**

Contributions are welcome! Please feel free to submit a pull request or open an issue to suggest improvements.

---

## **Contact**

For any inquiries or issues, please contact:  
[muneebkhan046@gmail.com](mailto:muneebkhan046@gmail.com)
