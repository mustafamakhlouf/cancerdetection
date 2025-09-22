# cancerdetection

# Skin Cancer Detection (Benign vs. Malignant)
  A deep learning project that detects skin cancer from dermoscopic images using EfficientNetB0 and provides explainability with Grad-CAM heatmaps.
  The project includes a Streamlit web app for easy interaction and visualization.

# Features
 Binary classification: Benign (BNN) vs Malignant (MAL)
 Powered by EfficientNetB0 backbone
 Grad-CAM heatmap for explainable AI
 Easy-to-use Streamlit UI
 Confidence score for predictions

# Project Structure
  ├── skin-cancer-binary (3).ipynb   # Jupyter Notebook (model training & experiments)
  ├── app.py                         # Streamlit web app for deployment
  ├── requirements.txt               # Python dependencies
  ├── EfficientNet_classifier_head.keras # Trained model weights
  └── README.md                      # Project documentation

# Installation & Setup
  1. Clone the repository
    git clone [https://github.com/your-username/skin-cancer-detection.gitcd](https://github.com/mustafamakhlouf/cancerdetection) skin-cancer-detection
  2. Create a virtual environment (optional but recommended)
    python -m venv venv
    source venv/bin/activate   # On Linux/Mac
    venv\Scripts\activate      # On Windows
  3.Install dependencies
    pip install -r requirements.txt
  4.Run the Streamlit app
    streamlit run app.py

# Usage

  1.Upload a dermoscopic image (.jpg, .jpeg, .png).
  2.Click Predict.
  3.View:
    .Predicted class: BNN or MAL
    .Confidence score
    ..Grad-CAM heatmap highlighting key regions

 # Model Details

  .Architecture: EfficientNetB0 (feature extractor) + custom dense classifier head
  .Input size: 224 × 224 × 3
  .Loss: Categorical Crossentropy
  .Metrics: Precision, Recall, F1, Binary Crossentropy, Categorical Accuracy
  .Explainability: Grad-CAM applied to top_conv layer
  
 # Results 
  Class	Precision	Recall	F1-Score
  Benign (BNN)	0.92	0.90	0.91
  Malignant (MAL)	0.88	0.91	0.89

 # Deployment Options
  Local: via streamlit run app.py

# Disclaimer
  This tool is for educational and research purposes only.
  It is NOT intended for medical use. Always consult a qualified dermatologist for diagnosis
  
