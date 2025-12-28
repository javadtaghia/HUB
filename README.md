#  [ICCV 2025] HUB: Holistic Unlearning Benchmark

Official Implementation of [Holistic Unlearning Benchmark: A Multi-Faceted Evaluation for Text-to-Image Diffusion Model Unlearning](https://arxiv.org/abs/2410.05664) (ICCV 2025)
- A comprehensive benchmark for evaluating unlearning methods in text-to-image diffusion models across multiple tasks and metrics
- 💡 Feel free to explore the code, open issues, or reach out for discussions and collaborations!

## 📦 Environment setup
### Installation
To set up the environment, follow these steps:
1. Clone the repository:
    ```
    git clone https://github.com/ml-postech/HUB.git
    cd HUB
    ```
2.	Create and activate the conda environment:
    ```bash
    conda create -n HUB python=3.10

    conda activate HUB
    # Install PyTorch separately (choose one):
    #  - Conda (recommended for CUDA):
    #      conda install -y pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
    #  - Pip (CUDA 11.8):
    #      pip install --index-url https://download.pytorch.org/whl/cu118 --extra-index-url https://pypi.org/simple "torch==2.7.1+cu118" torchvision torchaudio
    #  - CPU-only (pip):
    #      pip install torch==2.7.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

    # After installing PyTorch (e.g., with conda), install the remaining pip packages without dependencies
    # so pip does not attempt to reinstall PyTorch or conflicting CUDA wheels:
    pip install -r requirements.txt --no-deps
    ```

    ⚠️ **Note on binary package conflicts**:
    - Some packages (e.g., `opencv-python`, `pyarrow`) have strict NumPy requirements and may require specific NumPy versions (1.x vs 2.x). If you encounter errors like "compiled with NumPy 1.x", prefer installing these binaries via conda for compatibility.
    - Example (install with conda-forge):
      ```bash
      conda install -y numpy=1.26 pyarrow opencv -c conda-forge
      ```
    - Alternatively, pick a consistent NumPy major version (>=2 or <2) and install matching binary packages via conda or the appropriate wheel index.
    - If you want, I can recommend exact commands tailored to your GPU/OS.

### Download pre-trained models and datasets
- [Reference image dataset](https://huggingface.co/datasets/hi-sammy/HUB_reference_images)
    - To evaluate target proportion, reference images for each concept are required. We provide these reference images as part of a Hugging Face dataset.
    - Once downloaded, place the dataset under the `images/` directory:

- [Aesthetic Predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor)
  - For aesthetic score, we use the `sac+logos+ava1-l14-linearMSE.pth` model. 
  - Place it in the `/models/aesthetic_predictor` directory.

- [Q16](https://github.com/ml-research/Q16?tab=readme-ov-file)
    - Download `prompts.p` from [this link](https://drive.google.com/file/d/1lWKdUTvPDWY9hw7ruDdCHXMOqs24PbQq/view?usp=sharing) and place it at `/models/q16/` directory.

- [GIPHY Celebrity Detector](https://github.com/Giphy/celeb-detection-oss)
    - Download giphy_celeb_detector.zip from [this link](https://drive.google.com/file/d/1e1S4hDsqHkMBkSBSLAcuyLtpxhVFlbGg/view?usp=sharing) and extract it to `/models/` directory.



## 🖼️ Image generation
To perform evaluation using HUB, you must first generate images for each concept and task with your unlearned model. Use the prompts described below to generate images.

```
python source/image_generation.py \
    --method YOUR_METHOD \
    --target TARGET \
    --task TASK
```

`TASK` must be one of the following: `target_image`, `general_image`, `selective_alignment`, `pinpoint_ness`, `multilingual_robustness`, `attack_robustness`, `incontext_ref_image`.


## 💬 Prompt generation
All prompts used in our experiments are provided in the `prompts/` directory.
You can also generate prompts for your own target using the following scripts.


### Target prompt generation (base prompts)
```
python source/prompt_generation/prompt.py \
  --target YOUR_TARGET \
  [--style] [--nsfw]
```
- Use `--style` for style-related targets
- Use `--nsfw` for NSFW-related targets

### Multilingual robustness
After generating the base prompts, create multilingual versions:
```
python source/prompt_generation/translate_prompt.py \
  --target YOUR_TARGET
```

### Pinpoint-ness
```
python source/prompt_generation/pinpoint_ness.py \
  --target YOUR_TARGET
  ```

### Selective alignment
```
python source/prompt_generation/selective_alignment.py \
  --target YOUR_TARGET \
  [--style]   # Add only if this is a style-related target
```

## 📊 Evaluation
### How to evaluate own model?
For now, we support the following seven unlearning methods: [SLD](https://arxiv.org/abs/2211.05105), [AC](https://arxiv.org/abs/2303.13516), [ESD](https://arxiv.org/abs/2303.07345), [UCE](https://arxiv.org/abs/2308.14761), [SA](https://arxiv.org/abs/2305.10120), [Receler](https://arxiv.org/abs/2311.17717), [MACE](https://arxiv.org/abs/2403.06135). To evaluate your own model, you need to modify `model.__init__.py` to include the loading of your custom model. We recommend that you place your model in `models/sd/YOUR_METHOD/`.

### Run the evaluation
To run the all tasks at once, execute the following command:
```bash
python main.py --method YOUR_METHOD --target TARGET
```

## 🎯 How to evaluate each task individually?
Running evaluation using `main.py` takes a long time, as it evaluates all tasks at once. To evaluate each task separately, follow these commands. In the following examples, replace the variables according to the settings you want to evaluate. Make sure to execute below command before evaluating each task.
```bash
export PYTHONPATH=$PYTHONPATH:YOUR_PROJECT_DIR
```

### Target proportion, multilingual robustness, and attack robustness
The evaluation code is configured to run separately for each concept type, because different classifiers are used. For the `target_proportion`, `multilingual_robustness`, and `attack_robustness` tasks, run the following code.

* **Celebrity**
```
python source/eval/eval_gcd.py \
    --task TASK \
    --method YOUR_METHOD \
    --target TARGET
```

* **Style, IP (VLM)**

```
python source/eval/eval_vlm.py \
    --task TASK \
    --method YOUR_METHOD \
    --target TARGET
```

* **NSFW**

```
python source/eval/eval_nsfw.py \
    --task TASK \
    --method YOUR_METHOD \
    --target TARGET
```

### Quality & Alignment
- TASK: `general_image`, `target_image`.
- METRIC: `aesthetic`, `ImageReward`, `PickScore`, `FID`, `FID_SD`.

```
python source/quality/evaluation.py \
    --method YOUR_METHOD \
    --target TARGET \
    --task TASK \
    --metric METRIC
```

### Selective alignment

```
python source/eval/eval_selective_alignment.py \
    --method YOUR_METHOD \
    --target TARGET
```

### Pinpoint-ness

```
python source/eval/eval_vlm.py \
    --task "pinpoint_ness" \
    --method YOUR_METHOD \
    --target TARGET
```

## 📌 To Do
- [ ] Add two attacks.
- [ ] Add a leaderboard for each task.
- [ ] Add new unlearning methods.


## 📚 Citation
    @article{moon2024holistic,
        title={Holistic Unlearning Benchmark: A Multi-Faceted Evaluation for Text-to-Image Diffusion Model Unlearning},
        author={Moon, Saemi and Lee, Minjong and Park, Sangdon and Kim, Dongwoo},
        journal={arXiv preprint arXiv:2410.05664},
        year={2024}
    }


===

Let me prepare a short, precise step-by-step guide to install the NVIDIA driver on Ubuntu and some diagnostic commands to run; please run the checks I list and paste outputs if anything fails.## NVIDIA driver install guide for Ubuntu 🔧🐧

Below are the checks and the simplest, safest install steps. Run the diagnostic commands first and paste their outputs if something fails — I’ll give exact fixes.

1) Run these diagnostic checks (paste outputs) 🧾
```
lsb_release -a
uname -r
lspci -nn | grep -i nvidia || echo "No NVIDIA devices found"
sudo ubuntu-drivers devices
which nvidia-smi && nvidia-smi || echo "nvidia-smi not found / driver not active"
```
2) Recommended quick install (works for most Ubuntu systems) ✅
This uses Ubuntu's driver tooling and the graphics-drivers PPA:
```
sudo apt update
sudo apt install -y build-essential dkms linux-headers-$(uname -r) software-properties-common
sudo add-apt-repository -y ppa:graphics-drivers/ppa
sudo apt update
sudo ubuntu-drivers autoinstall
sudo reboot
```

This installs the driver Ubuntu recommends for your GPU and kernel.
If you prefer a specific driver (example: 535): sudo apt install -y nvidia-driver-535
3) Verify installation ✅
After reboot:
```
nvidia-smi
lsmod | grep nvidia
```
nvidia-smi should show your GPU and driver version.
If you plan to use CUDA/PyTorch: check python -c "import torch; print(torch.cuda.is_available())" after installing compatible PyTorch.
4) Secure Boot & module signing ⚠️
If Secure Boot is enabled, the kernel may block NVIDIA modules.
Check: sudo apt install -y mokutil && sudo mokutil --sb-state
If enabled, either disable Secure Boot in your BIOS/UEFI or follow Ubuntu’s module-signing/MOK enrollment steps (I can give instructions if needed).
5) Common troubleshooting tips 🛠️
If linux-headers-$(uname -r) fails: run sudo apt install linux-headers-generic and reboot, then retry driver install.
If ubuntu-drivers devices shows “no drivers”: confirm GPU presence with lspci.
If driver install fails, paste the last apt output or dmesg | grep -i nvidia so I can analyze it.
6) CUDA / PyTorch notes 💡
If you need CUDA for PyTorch, use the driver version required by the CUDA toolkit you want; I can give exact install commands (conda or apt/pip) once you tell me your Ubuntu version and whether you want CUDA system-wide or via conda.
If you want, run the diagnostic commands and paste the outputs here and I’ll give the exact install commands tailored to your Ubuntu/GPU and walk you through any error you hit. ✅

# torch is platform/CUDA-specific: install via conda or PyTorch site
# torch==2.7.1+cu118
torch==2.7.1  # (install +cu118 via pip index or use conda)


python source/image_generation.py \
  --method sd \
  --target "Nudity" \
  --task pinpoint_ness \
  --device cpu \ 

models:
Those names are **shorthand labels for different Stable Diffusion model variants or unlearning / safety techniques**, not official Hugging Face class names. You usually map them to **different checkpoints or training methods** and then load them with the same Diffusers API.



---

### **sd**

**Stable Diffusion (baseline)**
The original, unmodified model.

* No safety or unlearning applied
* Used as the reference model
* Example base: Stable Diffusion

---

### **esd**

**Erased Stable Diffusion**
A model where **specific concepts are erased** (for example artists, objects, styles).

* Uses concept-erasure training
* Often based on gradient projection or feature suppression
* Popular in “artist removal” papers

---

### **uce**

**Unified Concept Editing**
A method to **edit, suppress, or enhance concepts** in diffusion models.

* More flexible than ESD
* Can erase, weaken, or amplify concepts
* Often prompt- or latent-guided

---

### **salun**

**Safe and Accurate Latent UNlearning**
A structured unlearning method that removes concepts **while preserving non-target behavior**.

* Works in latent space
* Emphasizes retain vs forget balance
* Known from the SALUN paper and benchmarks
* SALUN

---

### **ac**

**Adversarial Conditioning**
Uses adversarial objectives to prevent generation of a target concept.

* Instead of erasing weights directly, it disrupts conditioning
* Often harder to detect but less surgical
* Can affect generation quality

---

### **sa**

**Safety-Aware / Self-Attention based suppression**
Usually refers to **attention manipulation**.

* Suppresses concepts via attention maps
* Non-destructive, reversible
* Sometimes implemented as inference-time masking

---

### **receler**

**Re-Centered Latent Erasure (or similar naming)**
A newer / less standardized term.

* Typically re-centers latent distributions to remove concept clusters
* Often experimental
* Name varies by repo

---

### **sld**

**Safe Latent Diffusion**
A safety-enhanced Stable Diffusion variant.

* Applies safety constraints in latent space
* Often compared with SALUN
* Tries to reduce harmful outputs without full erasure

---

### **mace**

**Model-Agnostic Concept Erasure**
One of the more formal and popular unlearning methods.

* Works across architectures
* Minimal retraining
* Strong preservation of unrelated concepts
* MACE

---

### **Important clarification**

These **are not loaded like this**:

```python
StableDiffusionPipeline.from_pretrained("mace")
```

Instead, they usually map to:

* Different **checkpoints**
* Different **fine-tuned UNet weights**
* Different **training recipes**

Example:

```python
pipe = StableDiffusionPipeline.from_pretrained(
    "your-org/sd-v1-5-mace-cat-erased",
    torch_dtype=torch.float16
)
```

---

### **Mental model**

Think of them as **labels for how the model was trained**, not model classes:

```
Stable Diffusion
 ├─ ESD  (hard erase)
 ├─ MACE (surgical erase)
 ├─ SALUN (balanced unlearning)
 ├─ SLD (safety-first)
 └─ AC / SA (conditioning & attention tricks)
```

