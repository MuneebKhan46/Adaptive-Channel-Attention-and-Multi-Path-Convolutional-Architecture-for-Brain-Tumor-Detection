Use the following commands to clone the repository
git clone https://github.com/yourusername/Adaptive-Channel-Attention-for-Brain-Tumor-Detection.git
cd Adaptive-Channel-Attention-for-Brain-Tumor-Detection


**Install Dependencies**
Make sure you have Python 3.x installed. You can install all the required dependencies with:
pip install -r requirements.txt

1. TensorFlow 2.12.0
2. NumPy
3. Pandas
4. Matplotlib
5. Seaborn
6. Scikit-learn
7. Pillow
   


**Dataset Setup**
This project uses the Brain Tumor MRI Dataset from Kaggle:
Brain Tumor MRI Dataset Link: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

Brain-tumor-mri-dataset/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── pituitary/
│   └── no_tumor/
└── Testing/

**Training and Testing**
python train.py
python metrics.py
python plots.py


