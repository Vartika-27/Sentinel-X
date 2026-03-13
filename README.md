# 🛡️ Sentinel-X

### Adversarial Machine Learning Demonstration Platform

Sentinel-X is an interactive platform that demonstrates how **adversarial attacks can manipulate deep learning models**.

The system allows users to upload an image, run adversarial attacks, and observe how model predictions change under small perturbations.

This project illustrates the vulnerability of modern neural networks and provides a visual, interactive way to understand adversarial machine learning.

---

# 🚀 Features

### 🧠 Deep Learning Model

* **ResNet-18** image classifier
* Pretrained on **ImageNet**
* PyTorch implementation

### ⚔️ Adversarial Attacks

* **FGSM — Fast Gradient Sign Method**
* **PGD — Projected Gradient Descent**

### 🖥️ Interactive Demo

* Upload any image
* Adjust attack strength (ε)
* Run adversarial attacks
* Compare predictions side-by-side

### 📊 Attack Analysis

The interface compares:

* Original prediction
* FGSM adversarial prediction
* PGD adversarial prediction

This helps visualize how adversarial perturbations affect model confidence.

---

# 🧱 System Architecture

```
Frontend (HTML + JavaScript)
        │
        │ HTTP Requests
        ▼
FastAPI Backend
        │
        │ PyTorch Model
        ▼
ResNet18 + Adversarial Attacks
```

---

# 📂 Project Structure

```
Sentinel-X
│
├── backend
│   ├── main.py          # FastAPI server
│   ├── attack.py        # FGSM + PGD attack implementations
│   ├── model.py         # Model loading
│   └── utils.py         # Image preprocessing + inference
│
├── frontend
│   └── index.html       # Interactive UI
│
└── README.md
```

---

# ⚙️ Backend Setup

Navigate to the backend directory:

```
cd backend
```

Install dependencies:

```
pip install torch torchvision fastapi uvicorn pillow
```

Run the API server:

```
uvicorn main:app --reload
```

Server will start at:

```
http://127.0.0.1:8000
```

---

# 🌐 Frontend Setup

Navigate to the frontend directory:

```
cd frontend
```

Run a local server:

```
python3 -m http.server 5500
```

Open the interface in your browser:

```
http://localhost:5500
```

---

# 🔬 Implemented Attacks

## FGSM (Fast Gradient Sign Method)

A single-step gradient-based attack that perturbs the input image by moving it in the direction that maximizes classification loss.

Formula:

```
x_adv = x + ε · sign(∇x J(θ, x, y))
```

Characteristics:

* Fast
* One-step attack
* Demonstrates basic adversarial vulnerability

---

## PGD (Projected Gradient Descent)

An iterative extension of FGSM that repeatedly perturbs the input while constraining the perturbation within an ε-ball.

Characteristics:

* Stronger attack
* Multi-step gradient updates
* Commonly used for adversarial robustness evaluation

---

# 🧪 Demo Workflow

1️⃣ Upload an image
2️⃣ Choose attack strength (ε)
3️⃣ Run adversarial attack
4️⃣ Compare predictions

The system displays:

```
Original Prediction
FGSM Attack Result
PGD Attack Result
```

This allows users to analyze whether adversarial perturbations change the model's prediction.

---

# 🎯 Purpose of the Project

Sentinel-X demonstrates the **security risks of machine learning systems** by showing how tiny, often imperceptible perturbations can cause incorrect predictions.

The platform is intended for:

* Machine Learning education
* Security research demonstrations
* Understanding adversarial robustness

---

# 🛠️ Tech Stack

**Backend**

* Python
* FastAPI
* PyTorch
* TorchVision

**Frontend**

* HTML
* JavaScript
* CSS

---

# 👨‍💻 Author

Developed as a prototype adversarial machine learning demonstration system.

---

# ⭐ Future Improvements

Potential extensions:

* Real-time perturbation visualization
* Attack strength analysis graphs
* Adversarial defense methods
* Additional attacks (CW, DeepFool, BIM)
* Model robustness benchmarking

---

