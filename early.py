import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, Callback

# Page configuration
st.set_page_config(page_title="Early Stopping Demo", layout="wide")

st.title("⚡ Early Stopping in Machine Learning")
st.write("Train a Neural Network with Early Stopping and observe training vs. validation loss live.")

# Sidebar Controls
st.sidebar.header("1. Dataset Settings")
n_samples = st.sidebar.slider("Number of Samples", 1000, 30000, 5000, step=1000)
n_features = st.sidebar.slider("Number of Features", 10, 100, 20)

st.sidebar.header("2. Early Stopping Settings")
patience = st.sidebar.slider("Patience (Epochs to wait)", 1, 50, 5)
min_delta = st.sidebar.number_input("Min Delta (Improvement Threshold)", value=0.001, format="%.4f")
max_epochs = st.sidebar.slider("Max Epochs Allowed", 20, 300, 100)

# Custom Callback for Live Plotting in Streamlit
class StreamlitProgressCallback(Callback):
    def __init__(self, plot_spot, status_spot, max_epochs):
        super().__init__()
        self.plot_spot = plot_spot
        self.status_spot = status_spot
        self.max_epochs = max_epochs
        self.train_loss = []
        self.val_loss = []

    def on_epoch_end(self, epoch, logs=None):
        self.train_loss.append(logs.get("loss"))
        self.val_loss.append(logs.get("val_loss"))
        
        # Update progress text
        self.status_spot.markdown(
            f"**Epoch:** `{epoch + 1}/{self.max_epochs}` | "
            f"**Train Loss:** `{logs.get('loss'):.4f}` | "
            f"**Val Loss:** `{logs.get('val_loss'):.4f}`"
        )
        
        # Redraw Loss Curve (Auto-scales cleanly frame-by-frame)
        fig, ax = plt.subplots(figsize=(10, 5))
        epochs_range = range(1, len(self.train_loss) + 1)
        
        ax.plot(epochs_range, self.train_loss, label="Training Loss", color="#1f77b4", linewidth=2.5)
        ax.plot(epochs_range, self.val_loss, label="Validation Loss", color="#ff7f0e", linewidth=2.5)
        
        # Highlight best validation epoch so far
        best_idx = np.argmin(self.val_loss)
        ax.scatter(best_idx + 1, self.val_loss[best_idx], color="green", s=100, zorder=5, label="Best Checkpoint")
        
        ax.set_xlabel("Epochs", fontsize=11)
        ax.set_ylabel("Loss", fontsize=11)
        ax.set_title("Training vs Validation Loss (Live)", fontsize=13)
        ax.legend(loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.5)
        
        # Fit X-axis dynamically to current epoch count
        ax.set_xlim(1, max(2, len(self.train_loss)))
        
        self.plot_spot.pyplot(fig, use_container_width=True)
        plt.close(fig)

# Button to Trigger Training
if st.button("🚀 Train Model with Early Stopping", type="primary"):
    
    # 1. Generate Synthetic Dataset
    # Dynamically allocate informative & redundant features to prevent make_classification errors
    n_informative = max(2, int(n_features * 0.6))
    n_redundant = max(1, int(n_features * 0.2))
    
    X, y = make_classification(
        n_samples=n_samples, 
        n_features=n_features, 
        n_informative=n_informative, 
        n_redundant=n_redundant, 
        random_state=42
    )
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    # 2. Build Keras Model
    model = Sequential([
        Dense(64, activation='relu', input_shape=(n_features,)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # 3. Setup Early Stopping Callback
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        min_delta=min_delta,
        restore_best_weights=True,
        verbose=1
    )
    
    # Placeholders for dynamic rendering
    status_spot = st.empty()
    plot_spot = st.empty()
    live_callback = StreamlitProgressCallback(plot_spot, status_spot, max_epochs)
    
    # 4. Train Model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=max_epochs,
        batch_size=64,
        callbacks=[early_stop, live_callback],
        verbose=0
    )
    
    stopped_epoch = len(history.history['loss'])
    best_epoch = np.argmin(history.history['val_loss']) + 1
    
    st.success(f"✅ Training completed! Early stopping halted training at epoch **{stopped_epoch}**.")
    st.info(f"🏆 Best validation loss achieved at **Epoch {best_epoch}** (Restored best weights).")