import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# layers that define a new shape and get their own block in the diagram
BLOCK_KIND = {
    "Conv2D": "grid",
    "MaxPool2D": "grid",
    "Dense": "column",
}
# layers that don't change shape -- folded into the label of the connecting arrow
SKIP_LAYERS = {"ReLU", "Flatten"}
# of those, which ones actually get a text label on the arrow (Flatten is implied
# by the grid->column transition and just adds visual clutter, so it's silent)
LABELED_SKIP_LAYERS = {"ReLU"}
SHORT_NAME = {"Conv2D": "Conv", "MaxPool2D": "Pool", "Dense": "FC"}


def _collect_blocks(model, input_shape):
    """
    Runs a dummy forward pass through model.layers to discover each layer's
    output shape. NOTE: this overwrites every layer's cached forward state
    (self.x, col_x, argmax_cache, ...) with values from the dummy input --
    harmless for a fresh/idle model, but don't call this between a real
    forward() and the matching backward() call, or the cache backward()
    relies on will have been clobbered.

    input_shape: (C, H, W) for a single sample.
    Returns:
        blocks: list of {"label": str, "shape": tuple, "kind": "grid"|"column"}
        arrow_labels: list of str, length len(blocks) - 1
    """
    x = np.zeros((1,) + input_shape)
    blocks = [{"label": "Input", "shape": input_shape, "kind": "grid"}]
    arrow_labels = []
    pending = []
    counters = {"Conv2D": 0, "MaxPool2D": 0, "Dense": 0}

    for layer in model.layers:
        kind = type(layer).__name__
        x = layer.forward(x)

        if kind in SKIP_LAYERS:
            if kind in LABELED_SKIP_LAYERS:
                pending.append(kind)
            continue

        counters[kind] += 1
        label = f"{SHORT_NAME[kind]}{counters[kind]}"
        blocks.append({"label": label, "shape": x.shape[1:], "kind": BLOCK_KIND[kind]})
        arrow_labels.append(" + ".join(pending))
        pending = []

    return blocks, arrow_labels


def _grid_block_size(shape, cell_size, tile_gap):
    C, H, W = shape
    cols = int(np.ceil(np.sqrt(C)))
    rows = int(np.ceil(C / cols))
    width = cols * W * cell_size + (cols - 1) * tile_gap
    height = rows * H * cell_size + (rows - 1) * tile_gap
    return width, height


def _column_block_size(shape, dot_size, dot_gap):
    (D,) = shape
    width = dot_size * 2
    height = (D - 1) * dot_gap + dot_size * 2
    return width, height


def _draw_grid_block(ax, x0, y_center, shape, cell_size, cell_gap, tile_gap):
    C, H, W = shape
    cols = int(np.ceil(np.sqrt(C)))
    width, height = _grid_block_size(shape, cell_size, tile_gap)
    y0 = y_center + height / 2

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
                        linewidth=0.3,
                        edgecolor="black",
                        facecolor="none",
                    )
                )
    return width, height


def _draw_column_block(ax, x0, y_center, shape, dot_size, dot_gap):
    (D,) = shape
    width, height = _column_block_size(shape, dot_size, dot_gap)
    y0 = y_center + height / 2 - dot_size

    for i in range(D):
        ax.add_patch(
            patches.Circle(
                (x0 + dot_size, y0 - i * dot_gap),
                dot_size,
                linewidth=0.3,
                edgecolor="black",
                facecolor="none",
            )
        )
    return width, height


def plot_architecture(
    model,
    input_shape=(1, 28, 28),
    save_path="img/architecture.png",
    cell_size=0.05,
    cell_gap=0.01,
    tile_gap=0.15,
    dot_size=0.03,
    dot_gap=0.09,
    block_gap=1.0,
):
    """
    Horizontal, left-to-right structural diagram of model.layers: conv/pool
    layers as tiled per-channel HxW grids, dense layers as columns of dots,
    ReLU/Flatten folded into labels on the connecting arrows. Pure structure
    -- every cell/dot is an empty outline, no real data plotted yet.
    """
    blocks, arrow_labels = _collect_blocks(model, input_shape)

    sizes = [
        (
            _grid_block_size(b["shape"], cell_size, tile_gap)
            if b["kind"] == "grid"
            else _column_block_size(b["shape"], dot_size, dot_gap)
        )
        for b in blocks
    ]

    fig, ax = plt.subplots(figsize=(22, 10))
    x_cursor = 0.0
    x_ranges = []

    for block, (width, height) in zip(blocks, sizes):
        if block["kind"] == "grid":
            _draw_grid_block(
                ax, x_cursor, 0.0, block["shape"], cell_size, cell_gap, tile_gap
            )
        else:
            _draw_column_block(ax, x_cursor, 0.0, block["shape"], dot_size, dot_gap)

        shape_str = "x".join(str(s) for s in block["shape"])
        ax.text(
            x_cursor + width / 2,
            height / 2 + 0.3,
            f"{block['label']}\n{shape_str}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

        x_ranges.append((x_cursor, x_cursor + width))
        x_cursor += width + block_gap

    for i, label in enumerate(arrow_labels):
        x_start, x_end = x_ranges[i][1], x_ranges[i + 1][0]
        ax.annotate(
            "",
            xy=(x_end, 0.0),
            xytext=(x_start, 0.0),
            arrowprops=dict(arrowstyle="->", linewidth=0.8),
        )
        if label:
            ax.text(
                (x_start + x_end) / 2,
                0.15,
                label,
                ha="center",
                va="bottom",
                fontsize=7,
                style="italic",
            )

    max_height = max(h for _, h in sizes)
    ax.set_xlim(-0.5, x_cursor)
    ax.set_ylim(-max_height / 2 - 1, max_height / 2 + 1)
    ax.set_aspect("equal")
    ax.axis("off")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    from model import Sequential
    from layers import Conv2D, ReLU, MaxPool2D, Flatten, Dense

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
    plot_architecture(model)
    print("Architecture diagram saved to img/architecture.png")
