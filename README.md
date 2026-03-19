# 🧵 Smart Fabric Defect Detection System (SFDDS)

> **GitHub Repository Description:**
> An automated, end-to-end deep learning pipeline for detecting textile defects using the ZJU-Leaper dataset. Includes highly optimized Kaggle notebooks for training and exporting YOLOv7, YOLOv11, and YOLOv26 models.

## 🖼️ Training Results & Inference

Below are the inference results obtained from our optimized models on the validation set. Defective areas are highlighted with red bounding boxes along with their confidence scores.

### YOLOv26 Detection Results
<p float="left">
  <img src="src/YOLO26/test 1 YOLO26.png" width="45%" />
  <img src="src/YOLO26/test 2 YOLO26.png" width="45%" />
</p>
<p float="left">
  <img src="src/YOLO26/test 3 YOLO26.png" width="45%" />
  <img src="src/YOLO26/test 4 YOLO26.png" width="45%" />
</p>

### YOLOv11 Detection Results
<p float="left">
  <img src="src/YOLOv11/test 1 YOLOv11.png" width="45%" />
  <img src="src/YOLOv11/test 2 YOLOv11.png" width="45%" />
</p>
<p float="left">
  <img src="src/YOLOv11/test 3 YOLOv11.png" width="45%" />
  <img src="src/YOLOv11/test 4 YOLOv11.png" width="45%" />
</p>


> *Note: YOLOv7 training inference visual results are currently pending and will be added soon.*

---

## 📌 Overview
This project provides a comprehensive and scalable solution for **Textile Defect Detection**, heavily optimized for the [ZJU-Leaper](https://github.com/bnhphan/ZJU-Leaper) dataset in a Kaggle environment. To ensure high accuracy and overcome raw data class imbalances, the pipeline automatically processes, balances, and augments the data before feeding it into state-of-the-art YOLO architectures.

Three separate Jupyter Notebooks (`.ipynb`) are provided to demonstrate the pipeline across different YOLO generations:
* **YOLOv7** (`YOLOv7_Textile_Defect_Detection.ipynb`)
* **YOLOv11** (`YOLOv11_Textile_Defect_Detection.ipynb` - powered by Ultralytics)
* **YOLOv26** (`YOLOv26_Textile_Defect_Detection.ipynb` - powered by Ultralytics)

## 📊 Dataset: ZJU-Leaper
The original ZJU-Leaper dataset contains approximately 75,000 textile images, but defective samples are significantly outnumbered by normal (defect-free) samples. 
Our preprocessing pipeline automatically solves this by:
- **Balancing the classes:** Randomly sampling up to `5,000` defective images and `5,000` normal images to construct a stable ~10k image dataset.
- **XML to YOLO Conversion:** Dynamically parses YOLO bounding box coordinates (`0-1` normalized format) from the original XML annotations.
- **Train/Val Split:** Uses a standard `80/20` split for robust training and validation.

## ⚙️ Hyperparameter Optimization
Each notebook has been carefully tuned to squeeze the maximum performance out of Kaggle's T4/P100 GPUs:
- **Epochs (`20`)**: Increased to allow the models sufficient time to learn from the expanded 10K image dataset.
- **Batch Size (`32`)**: Optimized to properly utilize GPU VRAM while maintaining fast and stable gradient updates.
- **Label Smoothing (`0.1`)**: Applied to prevent the models from becoming overconfident, thereby improving validation metrics.
- **Augmentation (YOLOv11 & YOLOv26)**: 
  - `mosaic=1.0` (Enabled 100%)
  - `mixup=0.15` (Enabled 15%)
  - `close_mosaic=5` (Disabled during the final 5 epochs to allow the network to fine-tune on realistic feature details).

## 🚀 Notebook Workflow
All three notebooks are structured into 6 clear stages that run autonomously on Kaggle:

1. **Environment Setup:** Installs dependencies (`ultralytics` for v11/v26, and custom legacy PyTorch 2.6 patches for `yolov7`).
2. **Data Preprocessing:** Executes the XML-to-YOLO conversion, generates subsets, and balances the defect classes.
3. **YAML Configuration:** Creates the `data.yaml` necessary for training.
4. **Optimized Training:** Starts the training process using the respective YOLO architectures and optimized hyperparameter profiles.
5. **Inference & Visualization:** Predicts bounding boxes on the validation set and visualizes the detected defects alongside confidence scores using `matplotlib`.
6. **ONNX Export:** Generates an `.onnx` version of the best PyTorch weights (`best.pt`) for cross-platform, local real-time deployment.

## 💻 How to Use (Kaggle)
1. Open [Kaggle](https://www.kaggle.com/) and create a new Notebook.
2. Add the **ZJU-Leaper** dataset to your Kaggle environment.
3. Upload one of the `YOLOv<version>_Textile_Defect_Detection.ipynb` files via the **File -> Import Notebook** option.
4. Set your Kaggle accelerator to **GPU T4 x2** or **P100**.
5. Click **Run All**. The model weights (`best.pt` and `best.onnx`) will be generated in your working directory.

## 🛠 Prerequisites for Local Usage
If you prefer running the code locally, you will need:
```bash
pip install ultralytics opencv-python matplotlib tqdm pyyaml torch torchvision onnx onnxruntime
```
*Note: Ensure your local environment has the dataset structured similarly to the Kaggle notebook paths (`/images` and `/xmls`).*
