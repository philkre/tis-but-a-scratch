"""
Per-epoch weight snapshots.

Analysis that asks "how did this change over training" needs the whole
trajectory, not just the best checkpoint, and wants to re-run without
retraining. Everything here writes to / reads from models/epochs/.
"""

import os

from nn import build_model, load_checkpoint, save_checkpoint

SNAPSHOT_DIR = "models/epochs"
EPOCHS = 12
LR = 0.1
BATCH_SIZE = 32
N_PROBE = 256


def snapshot_path(epoch):
    return f"{SNAPSHOT_DIR}/epoch_{epoch:02d}.npz"


def snapshot_callback():
    """epoch_callback for nn.train that saves every epoch, not just improvements."""
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    return lambda epoch, model: save_checkpoint(model, snapshot_path(epoch))


def load_snapshot(epoch, model=None):
    """Model with epoch's weights loaded. Reuses `model` if given, to avoid
    reallocating inside a loop over epochs."""
    model = model if model is not None else build_model()
    load_checkpoint(model, snapshot_path(epoch))
    return model


def require_snapshots():
    if not os.path.isdir(SNAPSHOT_DIR) or not os.listdir(SNAPSHOT_DIR):
        raise SystemExit(
            f"No snapshots in {SNAPSHOT_DIR}/. Run: python -m experiments.criticality"
        )
