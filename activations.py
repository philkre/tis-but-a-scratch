import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize

from visualize import (
    _grid_block_size,
    _column_block_size,
    BLOCK_KIND,
    SKIP_LAYERS,
    LABELED_SKIP_LAYERS,
    SHORT_NAME,
)


def _collect_activations(model, x):
    """
    Runs a real forward pass on a single example x, shape (1, C, H, W),
    recording each block's real output values (not just shape, unlike
    visualize._collect_blocks). Blocks immediately followed by a ReLU
    (Conv1, Conv2, FC1, FC2) record their POST-ReLU values; blocks with
    nothing following (Pool1, Pool2, and the final FC3 logits) record
    their own raw output.

    Intermediate Dense layers (every Dense except the last) are computed but
    don't get their own block -- a hidden dense layer's per-neuron activation
    has no spatial/positional meaning (neuron ordering is arbitrary), so a
    dot column doesn't add real interpretability value for a single example.
    Only the final Dense layer (the actual class logits, each with a fixed,
    known meaning) is drawn. They're still folded into the arrow label like
    ReLU/Flatten, so the diagram stays structurally honest about what ran.

    NOTE: mutates every layer's cached forward state, same caveat as
    visualize._collect_blocks -- don't call this between a real forward()
    and its matching backward().

    Returns:
        blocks: list of {"label": str, "values": ndarray, "kind": "grid"|"column"}
        arrow_labels: list of str, length len(blocks) - 1
    """
    layers = model.layers
    dense_indices = [i for i, l in enumerate(layers) if type(l).__name__ == "Dense"]
    last_dense_idx = dense_indices[-1] if dense_indices else None

    blocks = [{"label": "Input", "values": x[0], "kind": "grid"}]
    arrow_labels = []
    pending = []
    counters = {"Conv2D": 0, "MaxPool2D": 0, "Dense": 0}

    out = x
    i = 0
    while i < len(layers):
        layer = layers[i]
        kind = type(layer).__name__

        if kind in SKIP_LAYERS:
            out = layer.forward(out)
            if kind in LABELED_SKIP_LAYERS:
                pending.append(kind)
            i += 1
            continue

        if kind == "Dense" and i != last_dense_idx:
            out = layer.forward(out)
            counters[kind] += 1
            pending.append(f"{SHORT_NAME[kind]}{counters[kind]}")
            if i + 1 < len(layers) and type(layers[i + 1]).__name__ == "ReLU":
                out = layers[i + 1].forward(out)
                pending.append("ReLU")
                i += 2
            else:
                i += 1
            continue

        out = layer.forward(out)
        relu_applied = False
        if i + 1 < len(layers) and type(layers[i + 1]).__name__ == "ReLU":
            out = layers[i + 1].forward(out)
            relu_applied = True

        counters[kind] += 1
        label = f"{SHORT_NAME[kind]}{counters[kind]}"
        blocks.append({"label": label, "values": out[0], "kind": BLOCK_KIND[kind]})

        parts = pending + (["ReLU"] if relu_applied else [])
        arrow_labels.append(" + ".join(parts))
        pending = []

        i += 2 if relu_applied else 1

    return blocks, arrow_labels


def _normalize(values):
    vmin, vmax = float(values.min()), float(values.max())
    if vmax - vmin < 1e-12:
        vmax = vmin + 1e-12
    return Normalize(vmin=vmin, vmax=vmax)


def _draw_grid_block_colored(ax, x0, y_center, values, cell_size, cell_gap, tile_gap, cmap):
    C, H, W = values.shape
    cols = int(np.ceil(np.sqrt(C)))
    width, height = _grid_block_size(values.shape, cell_size, tile_gap)
    y0 = y_center + height / 2
    norm = _normalize(values)

    for c in range(C):
        tc, tr = c % cols, c // cols
        tile_x0 = x0 + tc * (W * cell_size + tile_gap)
        tile_y0 = y0 - tr * (H * cell_size + tile_gap)
        for i in range(H):
            for j in range(W):
                ax.add_patch(
                    patches.Rectangle(
                        (tile_x0 + j * cell_size, tile_y0 - (i + 1) * cell_size),
                        cell_size - cell_gap,
                        cell_size - cell_gap,
                        linewidth=0.1,
                        edgecolor="black",
                        facecolor=cmap(norm(values[c, i, j])),
                    )
                )
    return width, height


def _draw_column_block_colored(ax, x0, y_center, values, dot_size, dot_gap, cmap):
    (D,) = values.shape
    width, height = _column_block_size(values.shape, dot_size, dot_gap)
    y0 = y_center + height / 2 - dot_size
    norm = _normalize(values)

    for i in range(D):
        ax.add_patch(
            patches.Circle(
                (x0 + dot_size, y0 - i * dot_gap),
                dot_size,
                linewidth=0.1,
                edgecolor="black",
                facecolor=cmap(norm(values[i])),
            )
        )
    return width, height


def _draw_activation_row(
    ax, blocks, arrow_labels, y_offset,
    cell_size, cell_gap, tile_gap, dot_size, dot_gap, block_gap, cmap,
    row_label=None, x_offset=0.0,
):
    """Draws one example's full activation row, vertically centered at y_offset
    and horizontally starting at x_offset. Returns (total_width, row_height)."""
    sizes = [
        (
            _grid_block_size(b["values"].shape, cell_size, tile_gap)
            if b["kind"] == "grid"
            else _column_block_size(b["values"].shape, dot_size, dot_gap)
        )
        for b in blocks
    ]

    x_cursor = x_offset
    x_ranges = []

    for block, (width, height) in zip(blocks, sizes):
        if block["kind"] == "grid":
            _draw_grid_block_colored(
                ax, x_cursor, y_offset, block["values"], cell_size, cell_gap, tile_gap, cmap
            )
        else:
            _draw_column_block_colored(
                ax, x_cursor, y_offset, block["values"], dot_size, dot_gap, cmap
            )

        shape_str = "x".join(str(s) for s in block["values"].shape)
        ax.text(
            x_cursor + width / 2, y_offset + height / 2 + 0.3, f"{block['label']}\n{shape_str}",
            ha="center", va="bottom", fontsize=6,
        )

        x_ranges.append((x_cursor, x_cursor + width))
        x_cursor += width + block_gap

    for i, label in enumerate(arrow_labels):
        x_start, x_end = x_ranges[i][1], x_ranges[i + 1][0]
        ax.annotate(
            "", xy=(x_end, y_offset), xytext=(x_start, y_offset),
            arrowprops=dict(arrowstyle="->", linewidth=0.6),
        )
        if label:
            ax.text(
                (x_start + x_end) / 2, y_offset + 0.15, label,
                ha="center", va="bottom", fontsize=5, style="italic",
            )

    if row_label:
        ax.text(x_offset - 0.6, y_offset, row_label, ha="right", va="center", fontsize=10, fontweight="bold")

    row_height = max(h for _, h in sizes)
    return x_cursor - x_offset, row_height


def plot_activations(
    model, x, save_path="img/activations.png",
    cell_size=0.05, cell_gap=0.01, tile_gap=0.15,
    dot_size=0.03, dot_gap=0.09, block_gap=1.0,
    cmap_name="inferno",
):
    """
    Real activations for a single example x (shape (1, C, H, W)), drawn in
    the same block layout as visualize.plot_architecture, with cells colored
    by value instead of left as empty outlines. Each block is normalized
    independently (its own min/max for this example), so brightness is only
    comparable within a block, not across blocks.
    """
    blocks, arrow_labels = _collect_activations(model, x)
    cmap = plt.get_cmap(cmap_name)

    fig, ax = plt.subplots(figsize=(22, 10))
    width, height = _draw_activation_row(
        ax, blocks, arrow_labels, 0.0,
        cell_size, cell_gap, tile_gap, dot_size, dot_gap, block_gap, cmap,
    )

    ax.set_xlim(-0.5, width)
    ax.set_ylim(-height / 2 - 1, height / 2 + 1)
    ax.set_aspect("equal")
    ax.axis("off")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_activations_grid(
    model, X, y, save_path="img/activations_all_digits.png",
    cell_size=0.05, cell_gap=0.01, tile_gap=0.15,
    dot_size=0.03, dot_gap=0.09, block_gap=1.0, row_gap=1.5, col_gap=3.0,
    n_cols=2, cmap_name="inferno",
):
    """
    One example per digit 0-9 (first occurrence in X/y), each drawn as its
    own activation row (see plot_activations), arranged in an n_cols x
    (10 // n_cols) grid for side-by-side comparison. Column-major digit
    order: the first column gets digits 0..(rows_per_col-1), the next
    column gets the rest, and so on.
    """
    labels = np.argmax(y, axis=1)
    cmap = plt.get_cmap(cmap_name)
    rows_per_col = 10 // n_cols

    fig, ax = plt.subplots(figsize=(20 * n_cols, 3.2 * rows_per_col))

    x_offset = 0.0
    max_col_width = 0.0
    row_height = None
    min_y = 0.0

    for digit in range(10):
        row_in_col = digit % rows_per_col

        if row_in_col == 0 and digit > 0:
            x_offset += max_col_width + col_gap
            max_col_width = 0.0

        y_offset = -row_in_col * (row_height + row_gap) if row_height else 0.0

        idx = np.where(labels == digit)[0][0]
        x = X[idx : idx + 1]
        blocks, arrow_labels = _collect_activations(model, x)

        width, height = _draw_activation_row(
            ax, blocks, arrow_labels, y_offset,
            cell_size, cell_gap, tile_gap, dot_size, dot_gap, block_gap, cmap,
            row_label=f"Digit {digit}", x_offset=x_offset,
        )
        max_col_width = max(max_col_width, width)
        if row_height is None:
            row_height = height
        min_y = min(min_y, y_offset - height / 2)

    total_width = x_offset + max_col_width
    ax.set_xlim(-2, total_width)
    ax.set_ylim(min_y - 1, row_height / 2 + 1)
    ax.set_aspect("equal")
    ax.axis("off")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_filters(model, save_path="img/filters.png"):
    """
    Raw weights of the first Conv2D layer in model.layers, one grayscale
    image per filter, each normalized independently by matplotlib's default
    per-image scaling. No forward pass or example image needed -- these are
    fixed weights, not activations.
    """
    conv1 = next(layer for layer in model.layers if type(layer).__name__ == "Conv2D")
    W = conv1.params["W"]  # (out_channels, in_channels, kh, kw)
    out_channels = W.shape[0]

    fig, axes = plt.subplots(1, out_channels, figsize=(2 * out_channels, 2))
    if out_channels == 1:
        axes = [axes]

    for c in range(out_channels):
        filt = W[c, 0]  # (kh, kw) -- assumes in_channels == 1, true for the first conv layer
        axes[c].imshow(filt, cmap="gray")
        axes[c].set_title(f"Filter {c}", fontsize=8)
        axes[c].axis("off")

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    from data import load_mnist
    from model import Sequential
    from layers import Conv2D, ReLU, MaxPool2D, Flatten, Dense
    from train import load_checkpoint

    model = Sequential(
        [
            Conv2D(1, 6, 5),
            ReLU(),
            MaxPool2D(2, 2),
            Conv2D(6, 16, 5),
            ReLU(),
            MaxPool2D(2, 2),
            Flatten(),
            Dense(256, 120),
            ReLU(),
            Dense(120, 84),
            ReLU(),
            Dense(84, 10),
        ]
    )
    load_checkpoint(model, "models/best_model.npz")

    x_train, y_train, x_val, y_val, x_test, y_test = load_mnist()

    plot_filters(model)
    plot_activations(model, x_test[:1])
    plot_activations_grid(model, x_test, y_test)
    print("Saved img/filters.png, img/activations.png, img/activations_all_digits.png")
