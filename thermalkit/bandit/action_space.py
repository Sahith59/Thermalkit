import numpy as np

# The 4 actions the bandit can choose from.
# Higher batch_size = more GPU parallelism = more tok/s but more heat.
BATCH_SIZES = [1, 2, 4, 8]
N_ACTIONS = len(BATCH_SIZES)


def action_index(batch_size: int) -> int:
    return BATCH_SIZES.index(batch_size)


def batch_size_for(action_idx: int) -> int:
    return BATCH_SIZES[action_idx]


def feature_vector(action_idx: int, state: np.ndarray) -> np.ndarray:
    """Concatenate state with a one-hot action encoding.

    Shape: (state_dim + N_ACTIONS,) = (6 + 4,) = (10,)

    Disjoint LinUCB only uses the state, but we build the full feature
    here so the same function works for hybrid LinUCB later if needed.
    """
    one_hot = np.zeros(N_ACTIONS, dtype=np.float32)
    one_hot[action_idx] = 1.0
    return np.concatenate([state, one_hot])
