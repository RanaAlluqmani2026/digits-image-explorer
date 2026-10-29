import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Digits Image Explorer", layout="wide")


@st.cache_data
def load_data():
    digits = load_digits()
    return digits.images, digits.target


@st.cache_resource
def train_model():
    images, labels = load_data()
    indices = np.arange(len(labels))
    train_ids, test_ids = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=labels
    )
    model = LogisticRegression(max_iter=2000)
    model.fit(images[train_ids].reshape(-1, 64) / 16, labels[train_ids])
    accuracy = model.score(
        images[test_ids].reshape(-1, 64) / 16, labels[test_ids]
    )
    return model, test_ids, accuracy


def show_plot(fig):
    st.pyplot(fig)
    plt.close(fig)


images, labels = load_data()

st.title("Digits Image Explorer")
st.write(
    "Explore handwritten digits through image montages, pixel intensity "
    "distributions, and occlusion saliency."
)
st.caption(
    "Source: scikit-learn digits dataset · 1,797 grayscale images · "
    "8 × 8 pixels · labels 0–9 · pixel values 0–16"
)

tab1, tab2, tab3 = st.tabs(
    ["Image montage", "Image structure", "Model explanation"]
)

with tab1:
    st.header("How does handwriting vary within a digit?")
    selected = st.selectbox("Digit to display", ["All"] + list(range(10)))
    count = st.slider("Number of images", 10, 50, 20, step=5)
    seed = st.number_input("Random seed", 0, 1000, 42)

    pool = np.arange(len(labels))
    if selected != "All":
        pool = pool[labels == selected]

    ids = np.random.default_rng(seed).choice(
        pool, size=min(count, len(pool)), replace=False
    )

    rows = int(np.ceil(len(ids) / 5))
    fig, axes = plt.subplots(rows, 5, figsize=(10, rows * 2))
    for ax in axes.flat:
        ax.axis("off")
    for ax, idx in zip(axes.flat, ids):
        ax.imshow(images[idx], cmap="gray", vmin=0, vmax=16,
                  interpolation="nearest")
        ax.set_title(f"Digit {labels[idx]} · #{idx}", fontsize=9)
    fig.tight_layout()
    show_plot(fig)

    st.write(
        "**Interpretation:** The montage shows variation in stroke thickness, "
        "slant, and shape among images with the same digit label. Some "
        "handwritten digits also resemble other classes, which can make "
        "classification challenging."
    )

    st.subheader("Class balance in the full dataset")
    counts = np.bincount(labels, minlength=10)
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.bar(range(10), counts, color="steelblue")
    ax.set(xlabel="Digit", ylabel="Number of images", xticks=range(10))
    show_plot(fig)
    st.caption(
        f"Class counts range from {counts.min()} to {counts.max()} images; "
        "the dataset is approximately balanced."
    )

with tab2:
    st.header("A grayscale image is a matrix of intensities")
    idx = st.slider("Image index", 0, len(images) - 1, 0)
    img = images[idx]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    plotted = axes[0].imshow(
        img, cmap="gray", vmin=0, vmax=16, interpolation="nearest"
    )
    axes[0].set_title(f"Image #{idx} · Digit {labels[idx]}")
    axes[0].set(xlabel="Column", ylabel="Row")
    fig.colorbar(plotted, ax=axes[0], label="Pixel intensity")

    axes[1].hist(
        img.ravel(), bins=np.arange(-0.5, 17.5, 1),
        color="steelblue", edgecolor="white"
    )
    axes[1].set(
        title="Pixel intensity distribution",
        xlabel="Intensity", ylabel="Number of pixels"
    )
    fig.tight_layout()
    show_plot(fig)

    st.write(
        "**Interpretation:** Each image has one grayscale channel. "
        "Values near 0 represent the dark background, while values near 16 "
        "represent bright handwriting strokes. The histogram summarizes "
        "brightness but does not preserve the positions of the pixels."
    )
    with st.expander("View the 8 × 8 pixel matrix"):
        st.dataframe(img.astype(int))

    st.subheader("Compare two classes")
    col1, col2 = st.columns(2)
    with col1:
        digit_a = st.selectbox("First digit", range(10), index=0)
    with col2:
        digit_b = st.selectbox("Second digit", range(10), index=1)

    fig, ax = plt.subplots(figsize=(9, 3))
    bins = np.arange(-0.5, 17.5, 1)
    for digit, color in [(digit_a, "steelblue"), (digit_b, "darkorange")]:
        ax.hist(
            images[labels == digit].ravel(),
            bins=bins, density=True, histtype="step",
            linewidth=2, label=f"Digit {digit}", color=color
        )
    ax.set(xlabel="Pixel intensity", ylabel="Proportion per intensity bin")
    ax.legend()
    show_plot(fig)
    st.caption(
        "Distributions use all pixels in each selected class and are normalized "
        "to support comparison. Similar histograms do not imply similar shapes."
    )

with tab3:
    st.header("Which pixels influence the model's prediction?")
    model, test_ids, accuracy = train_model()
    st.metric("Held-out test accuracy", f"{accuracy:.1%}")
    st.caption(
        "Logistic regression · stratified 80/20 train/test split · "
        "random_state=42. Only held-out test images are selectable below."
    )

    test_position = st.slider(
        "Test example", 0, len(test_ids) - 1, 0
    )
    idx = int(test_ids[test_position])
    img = images[idx]
    x = img.reshape(1, -1) / 16

    probabilities = model.predict_proba(x)[0]
    class_position = int(np.argmax(probabilities))
    predicted = int(model.classes_[class_position])
    original_probability = float(probabilities[class_position])

    patch = st.selectbox("Occlusion patch size", [1, 2, 3], index=1)
    importance = np.zeros((8, 8))
    coverage = np.zeros((8, 8))

    for row in range(9 - patch):
        for col in range(9 - patch):
            masked = img.copy()
            masked[row:row + patch, col:col + patch] = 0
            masked_probability = model.predict_proba(
                masked.reshape(1, -1) / 16
            )[0, class_position]
            drop = original_probability - masked_probability
            importance[row:row + patch, col:col + patch] += drop
            coverage[row:row + patch, col:col + patch] += 1

    importance /= coverage
    limit = max(float(np.abs(importance).max()), 0.001)

    st.write(
        f"**True label:** {labels[idx]} · **Prediction:** {predicted} · "
        f"**Predicted-class probability:** {original_probability:.1%}"
    )

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    axes[0].imshow(img, cmap="gray", vmin=0, vmax=16)
    axes[0].set_title("Original image")
    heatmap = axes[1].imshow(
        importance, cmap="RdBu_r", vmin=-limit, vmax=limit
    )
    axes[1].set_title("Occlusion saliency")
    axes[2].imshow(img, cmap="gray", vmin=0, vmax=16)
    axes[2].imshow(
        importance, cmap="RdBu_r", vmin=-limit, vmax=limit, alpha=0.6
    )
    axes[2].set_title("Saliency overlay")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(
        heatmap, ax=axes.tolist(), shrink=0.7,
        label="Mean drop in predicted-class probability"
    )
    show_plot(fig)

    strongest = np.unravel_index(np.argmax(importance), importance.shape)
    st.write(
        "**Interpretation:** Red regions indicate that masking nearby pixels "
        "reduced the probability of the original predicted class. Blue regions "
        "indicate that masking increased it. White regions had little effect. "
        "Overlapping patch effects are averaged at each pixel."
    )
    st.write(
        f"For this example, the largest average probability drop is "
        f"{importance[strongest]:.3f} at row {strongest[0]}, "
        f"column {strongest[1]} (indices start at 0)."
    )
    st.caption(
        "This is a local explanation of this model, not proof of how humans "
        "recognize digits. Results depend on patch size and the zero-valued "
        "mask; larger patches can also remove several strokes at once."
    )
