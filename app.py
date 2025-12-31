# app_cnn.py
# A *very simple* Streamlit app to demonstrate how a CNN works using MNIST (built-in Keras dataset).
# It will:
# 1) Load MNIST
# 2) Build a tiny CNN
# 3) Train it and show logs + history (loss/accuracy curves)
# 4) Evaluate on test set
# 5) Let you pick a random test image and see the model's prediction

import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

st.set_page_config(page_title="Simple CNN Demo (MNIST)", layout="wide")
st.title("🧠 Simple CNN Demo (MNIST)")

st.write(
    "This app trains a small Convolutional Neural Network (CNN) on the **MNIST** handwritten digits dataset "
    "and lets you test predictions on random images."
)

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Training Settings")
epochs = st.sidebar.slider("Epochs", 1, 10, 3)
batch_size = st.sidebar.selectbox("Batch size", [32, 64, 128], index=1)
learning_rate = st.sidebar.selectbox("Learning rate", [1e-4, 5e-4, 1e-3, 2e-3], index=2)

use_small_subset = st.sidebar.checkbox("Use small subset for faster demo", value=True)
subset_train = st.sidebar.slider("Train subset size", 2000, 20000, 6000, 1000, disabled=not use_small_subset)
subset_test = st.sidebar.slider("Test subset size", 500, 5000, 1500, 500, disabled=not use_small_subset)

st.sidebar.markdown("---")
seed = st.sidebar.number_input("Random seed", 0, 999999, 42, 1)

# -----------------------------
# Helper: cache data
# -----------------------------
@st.cache_data(show_spinner=False)
def load_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    return (x_train, y_train), (x_test, y_test)

# -----------------------------
# Helper: build model
# -----------------------------
def build_cnn(lr: float):
    # A small CNN:
    # - Conv detects local patterns (edges/curves)
    # - MaxPool shrinks the image while keeping important signals
    # - Dense layers decide which digit it is
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

# -----------------------------
# Load and preprocess data
# -----------------------------
(x_train, y_train), (x_test, y_test) = load_mnist()

# Normalize to [0,1] and add channel dimension (28,28) -> (28,28,1)
x_train = (x_train.astype("float32") / 255.0)[..., None]
x_test = (x_test.astype("float32") / 255.0)[..., None]

# Use a smaller subset for fast demonstration (optional)
rng = np.random.default_rng(int(seed))
if use_small_subset:
    train_idx = rng.choice(len(x_train), size=int(subset_train), replace=False)
    test_idx = rng.choice(len(x_test), size=int(subset_test), replace=False)
    x_train_s, y_train_s = x_train[train_idx], y_train[train_idx]
    x_test_s, y_test_s = x_test[test_idx], y_test[test_idx]
else:
    x_train_s, y_train_s = x_train, y_train
    x_test_s, y_test_s = x_test, y_test

st.subheader("1) Data preview")
c1, c2 = st.columns(2)
with c1:
    st.write(f"Train: **{len(x_train_s)}** images")
    st.write(f"Test: **{len(x_test_s)}** images")
with c2:
    st.write("An example image:")
    example_i = int(rng.integers(0, len(x_train_s)))
    fig = plt.figure()
    plt.imshow(x_train_s[example_i].squeeze(), cmap="gray")
    plt.title(f"Label: {y_train_s[example_i]}")
    plt.axis("off")
    st.pyplot(fig)

# -----------------------------
# Training button
# -----------------------------
st.subheader("2) Train the CNN")

if "model" not in st.session_state:
    st.session_state.model = None
if "history" not in st.session_state:
    st.session_state.history = None
if "train_logs" not in st.session_state:
    st.session_state.train_logs = ""

class StreamlitLogCallback(tf.keras.callbacks.Callback):
    # This callback collects training logs so we can show them in Streamlit.
    def on_train_begin(self, logs=None):
        st.session_state.train_logs = ""

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        msg = (
            f"Epoch {epoch+1}: "
            f"loss={logs.get('loss', np.nan):.4f}, "
            f"accuracy={logs.get('accuracy', np.nan):.4f}, "
            f"val_loss={logs.get('val_loss', np.nan):.4f}, "
            f"val_accuracy={logs.get('val_accuracy', np.nan):.4f}\n"
        )
        st.session_state.train_logs += msg

train_col, info_col = st.columns([1, 2])

with train_col:
    if st.button("🚀 Train / Retrain Model", use_container_width=True):
        with st.spinner("Training..."):
            tf.keras.utils.set_random_seed(int(seed))
            model = build_cnn(float(learning_rate))

            history = model.fit(
                x_train_s,
                y_train_s,
                validation_split=0.2,
                epochs=int(epochs),
                batch_size=int(batch_size),
                verbose=0,  # we will capture our own logs via callback
                callbacks=[StreamlitLogCallback()],
            )

            st.session_state.model = model
            st.session_state.history = history.history

with info_col:
    st.write(
        "- **Conv2D** learns small pattern detectors (like tiny filters).\n"
        "- **MaxPooling** shrinks the image while keeping strong features.\n"
        "- **Dense** layers combine learned features to decide the digit (0–9)."
    )

# Show logs and history if available
if st.session_state.model is not None and st.session_state.history is not None:
    st.subheader("3) Training logs + history")

    log_left, plot_right = st.columns([1, 2])

    with log_left:
        st.write("📋 Logs")
        st.text_area("Training output", st.session_state.train_logs, height=240)

        st.write("✅ Evaluate on test data")
        test_loss, test_acc = st.session_state.model.evaluate(x_test_s, y_test_s, verbose=0)
        st.metric("Test accuracy", f"{test_acc:.3f}")
        st.metric("Test loss", f"{test_loss:.3f}")

    with plot_right:
        hist = st.session_state.history

        # Plot loss
        fig1 = plt.figure()
        plt.plot(hist["loss"], label="train loss")
        plt.plot(hist["val_loss"], label="val loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        st.pyplot(fig1)

        # Plot accuracy
        fig2 = plt.figure()
        plt.plot(hist["accuracy"], label="train acc")
        plt.plot(hist["val_accuracy"], label="val acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        st.pyplot(fig2)

# -----------------------------
# Prediction section
# -----------------------------
st.subheader("4) Try predictions on random test images")

if st.session_state.model is None:
    st.info("Train the model first (click **Train / Retrain Model**) to enable predictions.")
    st.stop()

# Pick a pool of random images to choose from
if "candidate_indices" not in st.session_state or st.button("🔁 Refresh random images"):
    st.session_state.candidate_indices = rng.choice(len(x_test_s), size=12, replace=False).tolist()

candidate_indices = st.session_state.candidate_indices

# Let the user pick one of the random images
choice = st.selectbox(
    "Choose an image index from the random set",
    options=list(range(len(candidate_indices))),
    format_func=lambda i: f"Option {i+1} (test row #{candidate_indices[i]})"
)

idx = candidate_indices[int(choice)]
img = x_test_s[idx]
true_label = int(y_test_s[idx])

# Predict probabilities and class
probs = st.session_state.model.predict(img[None, ...], verbose=0)[0]
pred_label = int(np.argmax(probs))

# Display
cA, cB = st.columns([1, 1])

with cA:
    fig = plt.figure()
    plt.imshow(img.squeeze(), cmap="gray")
    plt.axis("off")
    plt.title(f"True label: {true_label}")
    st.pyplot(fig)

with cB:
    st.write("🤖 Model prediction")
    st.success(f"Predicted: **{pred_label}**")
    st.write("Probabilities (0–9):")
    # Show probabilities as a simple table
    prob_table = {str(i): float(probs[i]) for i in range(10)}
    st.dataframe(pd.DataFrame([prob_table]), use_container_width=True)

    # Also show as a bar chart (simple visualization)
    figp = plt.figure()
    plt.bar(list(range(10)), probs)
    plt.xlabel("Digit class")
    plt.ylabel("Probability")
    st.pyplot(figp)

st.caption(
    "Tip: Increase epochs for higher accuracy, or turn off the subset option to train on the full MNIST dataset."
)
