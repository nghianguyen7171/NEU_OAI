# Olympic AI 2025 - Computer Vision Competition

## 🏆 National Economics University (NEU) Olympic AI Competition

This repository contains the **Computer Vision (CV) Track** materials for the Olympic AI 2025 competition organized by National Economic University. The competition focuses on **Medical Image Segmentation with Explainable AI**.

## 📋 Competition Overview

### **Topic 3: Computer Vision - Medical Image Segmentation**
- **Task**: Lung segmentation from chest X-ray images using U-Net architecture
- **Dataset**: Chest X-ray Masks and Labels (processed version)
- **Focus**: Segmentation accuracy + Explainable AI (GradCAM)
- **Platforms**: Both Local and Kaggle environments supported

### **Competition Structure**
- **Duration**: 4 hours
- **Environment**: Local development or Kaggle platform
- **Evaluation**: Model performance (40%) + Code quality (30%) + Report (30%)
- **Target**: Dice Score > 0.85 on validation set

## 📁 Repository Structure

```
NEU_OAI/
├── 📄 README.md                    # This file
├── 📄 requirements.txt             # Python dependencies
├── 📁 Exam/                       # Competition materials
│   ├── Local/                     # Local environment version
│   │   ├── Exam_CVSeg_local.pdf
│   │   ├── [Exam]CV_Segmentation_local.ipynb
│   │   └── [Code-based]CV_Segmentation_local.ipynb
│   └── Kaggle/                    # Kaggle platform version
│       ├── Exam_CVSeg_Kaggle.pdf
│       ├── [Exam]CV_Segmentation_Kaggle.ipynb
│       └── [Code-based]CV_Segmentation_Kaggle.ipynb
├── 📁 chest-xray-masks-and-labels/ # Dataset (hidden in Git)
└── 📁 results/                    # Training outputs (hidden in Git)
    ├── models/
    ├── plots/
    ├── predictions/
    └── gradcam/
```

## 🎯 Competition Materials

### **For Participants**

#### **📖 Exam Instructions**
- **Local Version**: `Exam/Local/Exam_CVSeg_local.pdf`
- **Kaggle Version**: `Exam/Kaggle/Exam_CVSeg_Kaggle.pdf`

#### **📝 Exam Notebooks**
- **Local**: `Exam/Local/[Exam]CV_Segmentation_local.ipynb`
- **Kaggle**: `Exam/Kaggle/[Exam]CV_Segmentation_Kaggle.ipynb`

#### **💻 Code-based Notebooks**
- **Local**: `Exam/Local/[Code-based]CV_Segmentation_local.ipynb`
- **Kaggle**: `Exam/Kaggle/[Code-based]CV_Segmentation_Kaggle.ipynb`

### **Dataset Information**
- **Source**: [Chest X-ray Masks and Labels](https://www.kaggle.com/datasets/nikhilpandey360/chest-xray-masks-and-labels)
- **Processed Version**: 283 images (from 800 original)
- **Split**: 226 train (80%) / 57 test (20%)
- **Train/Val**: 203 train (90%) / 23 validation (10%)
- **Size**: 256×256 pixels (resized from 3000×2919)
- **Format**: PNG (RGB images + grayscale masks)

## 🚀 Quick Start

### **Option 1: Local Environment**
```bash
# Clone repository
git clone https://github.com/nghianguyen7171/NEU_OAI.git
cd NEU_OAI

# Install dependencies
pip install -r requirements.txt

# Open exam notebook
jupyter notebook "Exam/Local/[Exam]CV_Segmentation_local.ipynb"
```

### **Option 2: Kaggle Platform**
1. Go to [Kaggle.com](https://kaggle.com)
2. Create new notebook
3. Add dataset: "chest-xray-masks-and-labels-processed"
4. Enable GPU T4 x2
5. Copy code from `Exam/Kaggle/[Exam]CV_Segmentation_Kaggle.ipynb`

## 📊 Expected Results

### **Model Performance**
- **Architecture**: Simple U-Net (encoder from scratch)
- **Loss Function**: Combined Loss (Dice 70% + BCE 30%)
- **Optimizer**: Adam (lr=0.001)
- **Target Metrics**: Dice > 0.85, IoU > 0.80, F1 > 0.85

### **Deliverables**
1. **Trained Model**: U-Net checkpoint (.pth file)
2. **Visualizations**: 5 best prediction samples
3. **GradCAM**: XAI attention heatmaps
4. **Training Plots**: Loss, Dice, IoU, F1 curves
5. **Report**: PDF analysis (max 5 pages)

## 🏅 Competition Rules

### **Allowed**
- ✅ All provided libraries and dependencies
- ✅ Local GPU/CPU or Kaggle GPU
- ✅ Provided dataset only
- ✅ U-Net architecture (from scratch)

### **Not Allowed**
- ❌ External APIs or cloud services
- ❌ Pre-trained models or transfer learning
- ❌ LLM assistance (ChatGPT, Claude, etc.)
- ❌ External code repositories

### **Evaluation Criteria**
- **Model Performance (40%)**: Dice Score > 0.85
- **Code Quality (30%)**: Clean code, documentation, error handling
- **XAI Implementation (20%)**: GradCAM visualization quality
- **Report Quality (10%)**: Clear analysis and insights

## 📚 Technical Details

### **Model Architecture**
```python
# Simple U-Net Configuration
- Input: 3 channels (RGB)
- Output: 1 channel (binary mask)
- Encoder: 4 blocks (64, 128, 256, 512 filters)
- Bottleneck: 1024 filters
- Decoder: ConvTranspose2D + skip connections
- Activation: ReLU + Sigmoid
```

### **Training Configuration**
```python
# Hyperparameters
IMAGE_SIZE = 256
BATCH_SIZE = 4
NUM_EPOCHS = 10  # Quick experiment
LEARNING_RATE = 0.001
TRAIN_SPLIT = 0.9  # 90% train
VAL_SPLIT = 0.1    # 10% validation
```

### **Data Augmentation**
- Horizontal flip (p=0.5)
- Rotation (±15°)
- Brightness/Contrast adjustment (±20%)
- Gaussian noise (optional)

## 🎓 Learning Objectives

Participants will gain experience in:
- **Medical Image Segmentation**: U-Net architecture and implementation
- **Deep Learning**: PyTorch, loss functions, optimization
- **Computer Vision**: Image preprocessing, augmentation, evaluation
- **Explainable AI**: GradCAM for model interpretability
- **Research Skills**: Experimentation, visualization, reporting

## 📞 Support

For competition-related questions:
- **Repository**: [NEU_OAI](https://github.com/nghianguyen7171/NEU_OAI)
- **Dataset**: [Chest X-ray Masks and Labels](https://www.kaggle.com/datasets/nikhilpandey360/chest-xray-masks-and-labels)
- **Platform**: [Kaggle](https://kaggle.com)

## 📄 License

This project is created for educational purposes as part of the Olympic AI 2025 competition at National Economic University.

---

**Good luck to all participants! 🚀**
