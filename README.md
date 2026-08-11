# ⚡ Early Stopping Visualization in Neural Networks

An interactive **Streamlit** web application built with **TensorFlow / Keras** that visualizes training dynamics and **Early Stopping** in real-time. Watch training loss vs. validation loss update frame-by-frame as the neural network trains on a synthetic dataset.

---

## 🎯 Key Features

* **⚡ Real-Time Live Graph:** Watch the loss curves adapt live at every epoch instead of waiting for training to complete.
* **📊 Customizable Synthetic Dataset:** Scale dataset size dynamically from **1,000 to 30,000 rows** and **10 to 100 features**.
* **🛑 Adjustable Early Stopping Hyperparameters:** Tweak `Patience`, `Min Delta`, and `Max Epochs` on the fly to see how they affect training termination.
* **⭐ Best Checkpoint Tracking:** Automatically highlights the epoch with the lowest validation loss where model weights were restored.
* **📐 Dynamic Graph Auto-Scaling:** Automatically adjusts X and Y axes in real-time so loss curves remain clear and visible without awkward squishing.

---

## 🛠️ Tech Stack

* **Language:** Python 3.10+
* **Frontend / UI:** [Streamlit](https://streamlit.io/)
* **Deep Learning Framework:** [TensorFlow / Keras](https://www.tensorflow.org/)
* **Machine Learning & Utilities:** [Scikit-Learn](https://scikit-learn.org/), NumPy, Pandas
* **Data Visualization:** [Matplotlib](https://matplotlib.org/)

---

## 🚀 Quick Start & Installation

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git](https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git)
cd YOUR-REPOSITORY
