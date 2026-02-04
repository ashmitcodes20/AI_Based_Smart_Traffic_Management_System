# AI_Based_Smart_Traffic_Management_System

🚦 AI-Based Smart Traffic Management System
📌 Project Overview

Traffic congestion, inefficient signal control, and lack of real-time traffic intelligence are major challenges in modern cities. This project presents an AI-Based Smart Traffic Management System that uses computer vision and deep learning to analyze traffic conditions and generate intelligent decisions.

The system detects vehicles from traffic images using a YOLOv8 model, estimates traffic density, dynamically adjusts signal timings, and integrates Automatic Number Plate Recognition (ANPR) for advanced features such as theft detection, vehicle migration analysis, and sustainability assessment. The complete system is implemented in Python using Google Colab and validated through experiments, visualizations, and performance metrics.


✨ Key Features

🚗 Vehicle Detection & Classification using YOLOv8
📊 Traffic Density Estimation from detected vehicles
🚦 Dynamic Traffic Signal Timing based on real-time density
🔢 Automatic Number Plate Recognition (ANPR) using OCR
🚨 Stolen Vehicle Detection using simulated databases
🌍 Vehicle Migration Analysis based on license plate regions
🌱 Green Tax / EV Analysis (simulation-based sustainability insights)
📈 Performance Evaluation with accuracy, precision, recall, loss curves, and confusion matrix


🧠 System Architecture
The system follows a modular pipeline:

Traffic Images
      ↓
YOLOv8 Vehicle Detection
      ↓
Vehicle Count (Traffic Density)
      ↓
Dynamic Signal Timing Logic
      ↓
Traffic Signal Decision

Vehicle Images
      ↓
License Plate OCR (EasyOCR)
      ↓
Database Matching
      ↓
• Theft Detection
• Migration Analysis
• Green Tax / EV Analysis

This modular design allows easy extension to real-time video feeds and smart city deployments.



🗂️ Project Structure
AI-Based-Smart-Traffic-Management-System/
│
├── notebooks/
│   └── Smart_Traffic_Management_System.ipynb
│
├── datasets/
│   ├── traffic_dataset/
│   └── license_plate_dataset/
│
├── models/
│   └── yolov8_best.pt
│
├── results/
│   ├── graphs/
│   ├── confusion_matrix.png
│   └── predictions/
│
├── outputs/
│   ├── traffic_density.csv
│   ├── traffic_signal_decisions.csv
│   └── ocr_results.csv
│
├── README.md
└── requirements.txt


📊 Datasets Used

Traffic Vehicles Object Detection Dataset (Kaggle)
Used for training and evaluating vehicle detection and traffic density estimation.

Indian License Plates Dataset (Kaggle)
Used for implementing and validating the ANPR module.

Mock / Simulated Datasets
Used for theft detection, vehicle migration, and green tax analysis due to restricted access to real government databases.


⚙️ Technologies & Tools

Python
YOLOv8 (Ultralytics)
OpenCV
EasyOCR
Pandas & NumPy
Matplotlib
Google Colab (GPU-enabled)


📈 Results & Evaluation

Vehicle detection accuracy evaluated using mAP@50
Precision and recall analysis for detection reliability
Training convergence validated using loss vs epochs
Confusion matrix for class-wise performance
Visual results for inference on unseen traffic images
OCR performance evaluated using detection success rate


▶️ How to Run the Project

Open the notebook in Google Colab
Enable GPU runtime
Install dependencies from requirements.txt
Run notebook cells sequentially:
Dataset loading
Model training
Evaluation & visualization
Inference
OCR and smart feature modules


🔮 Future Enhancements

Real-time video-based traffic monitoring
Reinforcement learning for adaptive signal control
Integration with real RTO / police databases
Edge deployment on CCTV systems (Jetson, Raspberry Pi)
Cloud-based dashboard for traffic authorities


👤 Author

Ashmit A. Shingarwade
Computer Science Engineering Student
MIT ADT University
