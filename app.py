# app_rnn.py
# A simple Streamlit app that demonstrates an RNN/LSTM on the built-in Keras IMDB dataset.
# It shows: training logs, history plots, test evaluation, and lets you select a test review to predict.

import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

st.set_page_config(page_title="Simple RNN/LSTM Demo (IMDB)", layout="wide")
st.title("🧠 Simple RNN / LSTM Demo (IMDB Sentiment)")
st.write("Train an LSTM on IMDB movie reviews (0 = negative, 1 = positive) and try predictions on test reviews.")

# ----------------------------
# Sidebar settings
# ----------------------------
st.sidebar.header("Training Settings")
num_words = st.sidebar.selectbox("Vocabulary size (top words)", [5000, 10000, 20000], index=1)
max_len = st.sidebar.slider("Max sequence length (padding length)", 50, 400, 200, 10)
embed_dim = st.sidebar.selectbox("Embedding dimension", [16, 32, 64], index=1)
rnn_units = st.sidebar.selectbox("LSTM units", [32, 64, 128], index=1)
dropout = st.sidebar.slider("Dropout", 0.0, 0.6, 0.3, 0.05)

epochs = st.sidebar.slider("Epochs", 1, 10, 3)
batch_size = st.sidebar.selectbox("Batch size", [32, 64, 128], index=1)
learning_rate = st.sidebar.selectbox("Learning rate", [1e-4, 5e-4, 1e-3, 2e-3], index=2)

use_small_subset = st.sidebar.checkbox("Use smaller subset (faster demo)", value=True)
train_subset = st.sidebar.slider("Train subset size", 2000, 25000, 6000, 1000, disabled=not use_small_subset)
test_subset = st.sidebar.slider("Test subset size", 500, 25000, 3000, 500, disabled=not use_small_subset)

seed = st.sidebar.number_input("Random seed", 0, 999999, 42, 1)

st.sidebar.markdown("---")
st.sidebar.header("Prediction Settings")
words_to_show = st.sidebar.slider("Words to show (decoded)", 30, 400, 120, 10)

# ----------------------------
# Cache dataset loading
# ----------------------------
@st.cache_data(show_spinner=False)
def load_imdb(num_words: int):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=num_words)
    return (x_train, y_train), (x_test, y_test)

@st.cache_data(show_spinner=False)
def load_word_index():
    return tf.keras.datasets.imdb.get_word_index()

# ----------------------------
# Decode helpers
# ----------------------------
word_index = load_word_index()
id_to_word = {idx + 3: w for w, idx in word_index.items()}
id_to_word[0] = "<PAD>"
id_to_word[1] = "<START>"
id_to_word[2] = "<UNK>"
id_to_word[3] = "<UNUSED>"

def decode_review(review_ids):
    return " ".join(id_to_word.get(i, "<UNK>") for i in review_ids)

# ----------------------------
# Load data
# ----------------------------
(x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = load_imdb(int(num_words))

# Optional: use subset
rng = np.random.default_rng(int(seed))
if use_small_subset:
    train_idx = rng.choice(len(x_train_raw), size=int(train_subset), replace=False)
    test_idx = rng.choice(len(x_test_raw), size=int(test_subset), replace=False)
    x_train_raw = [x_train_raw[i] for i in train_idx]
    y_train_raw = y_train_raw[train_idx]
    x_test_raw = [x_test_raw[i] for i in test_idx]
    y_test_raw = y_test_raw[test_idx]

# Pad sequences
x_train = tf.keras.preprocessing.sequence.pad_sequences(
    x_train_raw, maxlen=int(max_len), padding="pre", truncating="pre"
)
x_test = tf.keras.preprocessing.sequence.pad_sequences(
    x_test_raw, maxlen=int(max_len), padding="pre", truncating="pre"
)

st.subheader("1) Data preview")
c1, c2, c3 = st.columns(3)
with c1:
    st.write(f"Train reviews: **{len(x_train)}**")
    st.write(f"Test reviews: **{len(x_test)}**")
with c2:
    st.write("Shapes")
    st.write(f"x_train: `{x_train.shape}`")
    st.write(f"y_train: `{y_train_raw.shape}`")
with c3:
    st.write("Label meaning")
    st.write("0 = negative 😞")
    st.write("1 = positive 😄")

# ----------------------------
# Build model
# ----------------------------
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(int(max_len),)),
        tf.keras.layers.Embedding(input_dim=int(num_words), output_dim=int(embed_dim)),
        tf.keras.layers.LSTM(int(rnn_units)),
        tf.keras.layers.Dropout(float(dropout)),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(learning_rate)),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

# ----------------------------
# Store in session state
# ----------------------------
if "model" not in st.session_state:
    st.session_state.model = None
if "history" not in st.session_state:
    st.session_state.history = None
if "train_logs" not in st.session_state:
    st.session_state.train_logs = ""

class StreamlitLogCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        st.session_state.train_logs = ""

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        msg = (
            f"Epoch {epoch+1}: "
            f"loss={logs.get('loss', np.nan):.4f}, "
            f"acc={logs.get('accuracy', np.nan):.4f}, "
            f"val_loss={logs.get('val_loss', np.nan):.4f}, "
            f"val_acc={logs.get('val_accuracy', np.nan):.4f}\n"
        )
        st.session_state.train_logs += msg

st.subheader("2) Train the RNN/LSTM")

left, right = st.columns([1, 2])

with left:
    if st.button("🚀 Train / Retrain Model", use_container_width=True):
        with st.spinner("Training..."):
            tf.keras.utils.set_random_seed(int(seed))
            model = build_model()
            history = model.fit(
                x_train, y_train_raw,
                epochs=int(epochs),
                batch_size=int(batch_size),
                validation_split=0.2,
                verbose=0,
                callbacks=[StreamlitLogCallback()],
            )
            st.session_state.model = model
            st.session_state.history = history.history

with right:
    st.write(
        "**How this works (simple):**\n"
        "- **Embedding** turns word IDs into meaning vectors.\n"
        "- **LSTM** reads the review in order and keeps memory.\n"
        "- **Sigmoid** outputs a probability of positive sentiment.\n"
    )

# Show model summary once trained
if st.session_state.model is not None:
    with st.expander("Show model summary"):
        s = []
        st.session_state.model.summary(print_fn=lambda x: s.append(x))
        st.code("\n".join(s))

# ----------------------------
# Training logs + history plots
# ----------------------------
if st.session_state.model is not None and st.session_state.history is not None:
    st.subheader("3) Logs, history, and evaluation")

    log_col, plot_col = st.columns([1, 2])

    with log_col:
        st.write("📋 Training logs")
        st.text_area("Logs", st.session_state.train_logs, height=240)

        test_loss, test_acc = st.session_state.model.evaluate(x_test, y_test_raw, verbose=0)
        st.metric("Test accuracy", f"{test_acc:.3f}")
        st.metric("Test loss", f"{test_loss:.3f}")

    with plot_col:
        hist = st.session_state.history

        fig1 = plt.figure()
        plt.plot(hist["loss"], label="train loss")
        plt.plot(hist["val_loss"], label="val loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        st.pyplot(fig1)

        fig2 = plt.figure()
        plt.plot(hist["accuracy"], label="train acc")
        plt.plot(hist["val_accuracy"], label="val acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        st.pyplot(fig2)

# ----------------------------
# Prediction section
# ----------------------------
st.subheader("4) Predict on test reviews")

if st.session_state.model is None:
    st.info("Train the model first to enable predictions.")
    st.stop()

# Create a random list of candidate reviews to pick from
if "candidates" not in st.session_state or st.button("🔁 Refresh random test reviews"):
    st.session_state.candidates = rng.choice(len(x_test_raw), size=10, replace=False).tolist()

candidates = st.session_state.candidates

choice = st.selectbox(
    "Pick a review (random selection)",
    options=list(range(len(candidates))),
    format_func=lambda i: f"Option {i+1} (test item #{candidates[i]})"
)

idx = candidates[int(choice)]
review_ids = x_test_raw[idx]
true_label = int(y_test_raw[idx])

# Pad it
review_padded = tf.keras.preprocessing.sequence.pad_sequences(
    [review_ids], maxlen=int(max_len), padding="pre", truncating="pre"
)

# Predict
p = float(st.session_state.model.predict(review_padded, verbose=0)[0][0])
pred_label = 1 if p >= 0.5 else 0

# Show results
colA, colB = st.columns([1, 1])

with colA:
    st.write("📝 Review (decoded)")
    text = decode_review(review_ids[:int(words_to_show)])
    st.text_area("Text", text, height=220)

with colB:
    st.write("🤖 Prediction")
    st.success(f"Predicted: **{'positive 😄' if pred_label==1 else 'negative 😞'}**")
    st.write(f"Probability of positive: **{p:.3f}**")
    st.write(f"True label: **{'positive 😄' if true_label==1 else 'negative 😞'}**")

    # Simple bar chart
    figp = plt.figure()
    plt.bar(["negative", "positive"], [1 - p, p])
    plt.ylim(0, 1)
    plt.ylabel("Probability")
    st.pyplot(figp)

st.caption(
    "Tip: Increase epochs for better accuracy. Smaller subsets are faster but less accurate."
)
