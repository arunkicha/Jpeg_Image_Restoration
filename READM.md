Quality-Aware JPEG Restoration Using a Cascaded Deep Network with Hybrid Spatial–Frequency Loss

**Abstract - JPEG compression frequently results in artifacts such as blocking, ringing, and the loss of fine textures, particularly at low-quality factors. These distortions can significantly diminish image clarity and adversely affect subsequent processing tasks. This project introduces a novel hybrid deep learning model that effectively integrates Quality-Factor Conditioning, multi-stage feature fusion, and frequency-based loss functions to restore high-quality images from heavily compressed JPEG inputs. Inspired by advancements in state-of-the-art SFP and multi-part networks, our approach employs spatial MAE, FFT loss, perceptual VGG loss, and evaluates structural consistency using SSIM-based metrics. The model has been trained on the DIV2K and Flickr2K datasets, with evaluations conducted on LIVE1, BSD500, and Urban100 datasets at quality factors ranging from 10 to 90. Furthermore, a Gradio-based demonstration interface has been developed to facilitate interactive visualization of restoration outcomes. The system is adept at reducing compression artifacts and enhancing perceptual quality while maintaining structural integrity.**

**Team Members:**

Akshadh Athreya 23MIA1065
Kishore GVN 23MIA1155
Arunkumaran A 23MIA1171

**Base Paper:**

W. Wang, X. Xie, L. Deng, S. Xu and Q. Zhang, “Artifact Reduction in JPEG-Compressed Images via Quality-Aware Deep Networks,” IEEE Transactions on Image Processing, vol. 33, pp. 1450–1463, 2024.


**Tools and Libraries Used**

Python 3.10
PyTorch 2.x (CUDA enabled)
torchvision
OpenCV
NumPy / Pandas
scikit-image
tqdm
Gradio (for demo UI)
Google Colab / NVIDIA A100 for training
YAML for configuration
Matplotlib (visualization)

**Steps to Execute the Code**
git clone https://github.com/your-repo/jpeg-restoration.git
cd jpeg-restoration

Install dependencies: pip install -r requirements.txt

Prepare Datasets: 
/content/datasets/JPEG_Restoration_Datasets/
   ├── DIV2K/train/HR
   ├── Flickr2K/train
   ├── LIVE1/color/qf_XX/
   ├── LIVE1/refimgs/
   ├── BSD500/color/qf_XX/
   ├── BSD500/refimgs/
   ├── Urban100/color/qf_XX/
   └── Urban100/refimgs/

Train the Model: python train.py --opt opts/opts_se_hybrid.yml

Test the Model: python test.py --input "<path_to_qf_folder>" --gt "<path_to_refimgs>" --ckpt "<path_to_best_model>" --output "results/" --metric y --tta --q 40

Run the demo: python app.py

**Dataset Description**

**Training Datasets**

1.DIV2K: 800 high-resolution images - Used for high-quality reconstruction learning

2.Flickr2K: 2650+ diverse natural images - Expands training variety and robustness

**Testing / Benchmark Datasets**

1. LIVE1: 29 natural images - Standard for JPEG restoration benchmarking

2. BSD500: 500 varied natural scenes - Evaluated using color versions (converted)

3. Urban100: 100 images with rich textures (buildings, patterns) - Measures performance on high-detail structures
