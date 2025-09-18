import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB0
from PIL import Image

# =========================
# Constants & Configuration
# =========================
IMG_SIZE = 224
MODEL_WEIGHTS_PATH = "EfficientNet_classifier_head.keras"

METRICS = [
    tf.keras.metrics.BinaryCrossentropy(name='binary_crossentropy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.F1Score(name='f1'),
    tf.keras.metrics.CategoricalAccuracy(name='categorical_accuracy'),
]

# =========================
# Model Building & Loading
# =========================
def build_model(num_classes, metrics=METRICS):
    inputs = tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model = EfficientNetB0(include_top=False, input_tensor=inputs, weights=None)
    base_model.trainable = False

    x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(base_model.output)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2, name="top_dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="pred")(x)

    model = tf.keras.Model(inputs, outputs, name="EfficientNet")
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=metrics)
    return model

@st.cache_resource
def load_trained_model():
    try:
        num_classes = 2
        model = build_model(num_classes=num_classes)
        model.load_weights(MODEL_WEIGHTS_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.error(f"Please ensure the weights file '{MODEL_WEIGHTS_PATH}' is in the correct directory.")
        return None

# =========================
# Grad-CAM Visualization
# =========================
def generate_gradcam(img_array, model, last_conv_layer_name="top_conv"):
    """
    Generates a Grad-CAM heatmap.
    """
    grad_model = Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_index = tf.argmax(predictions[0])
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(conv_outputs[0] * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= (tf.reduce_max(heatmap) + tf.keras.backend.epsilon())
    return heatmap.numpy()

def overlay_heatmap(heatmap, original_img, alpha=0.5, colormap=cv2.COLORMAP_JET):
    heatmap = cv2.resize(heatmap, (original_img.width, original_img.height))
    heatmap = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap, colormap)
    original_np = np.array(original_img.convert("RGB"))
    superimposed_img = cv2.addWeighted(original_np, 1 - alpha, heatmap_colored, alpha, 0)
    return superimposed_img

# =========================
# Streamlit User Interface
# =========================
st.set_page_config(page_title="Skin Cancer Detection", page_icon="🧠", layout="wide")
st.title("🧠 Skin Cancer Detection")
st.markdown("""
Upload a **dermoscopic image** to predict if it's **Benign (BNN)** or **Malignant (MAL)**.
The model will also generate a heatmap to show which parts of the image were most influential for the prediction.
""")

model = load_trained_model()

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        try:
            original_img = Image.open(uploaded_file).convert("RGB")
            st.image(original_img, caption="Uploaded Image", use_container_width=True)
        except Exception as e:
            st.error(f"Error opening image file: {e}")
            uploaded_file = None
    
    predict_button_disabled = not (uploaded_file and model)
    if st.button("🔍 Predict", disabled=predict_button_disabled, use_container_width=True):
        with st.spinner('Analyzing image... Please wait.'):
            img_resized = original_img.resize((IMG_SIZE, IMG_SIZE))
            img_array = image.img_to_array(img_resized)
            img_array_expanded = np.expand_dims(img_array, axis=0)
            img_array_preprocessed = preprocess_input(img_array_expanded)

            prediction = model.predict(img_array_preprocessed)[0]
            pred_idx = np.argmax(prediction)
            pred_label = "MAL" if pred_idx == 1 else "BNN"
            confidence = prediction[pred_idx]

            try:
                heatmap = generate_gradcam(img_array_preprocessed, model)
                cam_image = overlay_heatmap(heatmap, original_img)
            except Exception as e:
                st.error(f"Could not generate Grad-CAM heatmap: {e}")
                cam_image = None
        
        with col2:
            st.markdown("### **Prediction Results**")
            st.metric(label="Predicted Class", value=pred_label)
            st.metric(label="Confidence", value=f"{confidence:.2%}")
            if cam_image is not None:
                st.image(cam_image, caption="Grad-CAM Heatmap", use_container_width=True)
                st.info("""
                **About the Heatmap:** The red areas indicate the regions of the image
                that the model focused on most to make its prediction.
                """)

if not model:
    st.error("Model could not be loaded. The application cannot proceed.")