# Software Requirements Specification (SRS)

## 1. Introduction

### 1.1 Purpose
The purpose of this document is to define the software requirements for **Sentinel-X**, an interactive Adversarial Machine Learning Demonstration Platform. This document outlines the system's architecture, functional specifications, and non-functional requirements to provide a comprehensive understanding of its capabilities.

### 1.2 Scope
Sentinel-X is a web-based educational and demonstration tool designed to illustrate how adversarial attacks can manipulate deep learning model predictions. The system allows users to upload images, apply adversarial perturbations (FGSM and PGD) controlled by a customizable strength parameter (epsilon), and compare the network's classifications before and after the attack.

### 1.3 Target Audience
- Machine Learning Students and Educators
- Security Researchers
- AI Enthusiasts interested in adversarial robustness

---

## 2. Overall Description

### 2.1 Product Perspective
Sentinel-X operates as a client-server architecture consisting of a vanilla HTML/JS/CSS frontend and a Python FastAPI backend. The backend loads a pre-trained ResNet-18 model (from torchvision) to perform inference and generate adversarial examples.

### 2.2 Technology Stack
- **Frontend**: HTML5, Vanilla JavaScript, CSS3 (Custom Dark Mode UI)
- **Backend API**: Python, FastAPI, Uvicorn
- **Machine Learning**: PyTorch, TorchVision (ImageNet-pretrained ResNet-18)
- **Image Processing**: Pillow (PIL)

### 2.3 User Environment
Users interact with the system via a modern web browser. The backend runs locally or on a server capable of executing PyTorch models (CPU or GPU).

---

## 3. System Features & Functional Requirements

### 3.1 Image Upload Module
- **Req-3.1.1**: The system must allow users to upload images via a drag-and-drop interface or file selection dialog.
- **Req-3.1.2**: Supported file types must include JPEG, PNG, WebP, and BMP.
- **Req-3.1.3**: The system must validate the uploaded file format and raise an HTTP 415 error for unsupported media types.

### 3.2 Attack Configuration
- **Req-3.2.1**: The UI must provide a slider to adjust the attack strength parameter (`Epsilon` / `ε`).
- **Req-3.2.2**: The epsilon value must be configurable within a range of `0.005` to `0.3`, with step increments of `0.005`.
- **Req-3.2.3**: The default epsilon value should be `0.03`.

### 3.3 Inference & Attack Generation Pipeline
- **Req-3.3.1**: The backend must resize and preprocess incoming images to 224x224 pixels, normalizing them with ImageNet standard mean and standard deviation.
- **Req-3.3.2**: The backend must run inference on the original image using the ResNet-18 model to predict the class and calculate confidence.
- **Req-3.3.3**: The backend must implement the **Fast Gradient Sign Method (FGSM)** to generate a perturbed image and subsequently classify it to retrieve the new prediction and confidence.
- **Req-3.3.4**: The backend must implement **Projected Gradient Descent (PGD)** as a stronger iterative attack, generating a perturbed image and retrieving its classification.
- **Req-3.3.5**: The API payload must return the original filename, selected epsilon, and predictions (class integer index and float confidence) for the Original, FGSM, and PGD passes.

### 3.4 Attack Visualization
- **Req-3.4.1**: The backend must generate a visualization image representing the original image, adversarial perturbation noise, and the final adversarial image.
- **Req-3.4.2**: The frontend must retrieve and display this generated visualization using a cache-busting mechanism to ensure the latest image is shown per attack.

### 3.5 Results Evaluation & Verdict
- **Req-3.5.1**: The UI must display the classification indices and confidence percentages side-by-side for the original, FGSM, and PGD results.
- **Req-3.5.2**: The UI must calculate an attack "Verdict." If either FGSM or PGD changes the predicted class from the original, a success warning badge must be displayed. If the class remains the same, a "Model held" badge should be shown.

---

## 4. Non-Functional Requirements

### 4.1 Interface & Usability
- **UI Design**: The application must feature a "cyber" aesthetic using dark surface colors (`#080c10`), cyan/neon red accents (`#00ffe7`, `#ff3c6e`), monospaced and sans-serif typography (`Share Tech Mono`, `Syne`), and a subtle scanline background.
- **Responsiveness**: The UI layout (specifically the prediction grid) must collapse into a single-column layout on viewports smaller than 600px.
- **Feedback**: The system must provide visual loading states (e.g., animated submit button) during backend API calls to prevent duplicate submissions.

### 4.2 Performance & Reliability
- **Model Loading**: The FastAPI backend must utilize a lifespan context manager to load the ResNet-18 model weights exactly once upon application startup.
- **Error Handling**: The backend must handle decoding errors (HTTP 422) and inference failures (HTTP 500) gracefully, and the frontend must surface these to the user via an error box.
- **Concurrency**: The API should be asynchronous (`async def`) for route handling, though model inferences are synchronously computed.

### 4.3 Security & Integration
- **CORS**: The backend must enable Cross-Origin Resource Sharing (CORS) allowing all origins to facilitate local frontend-backend communication.
- **Static Assets**: The backend must mount the root directory as a static file server to serve the generated `attack_visualization.png`.
