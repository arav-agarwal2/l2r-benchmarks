from src.buffers.SimpleReplayBuffer import ReplayBuffer
import numpy as np


def test_buffer():
    buffer = ReplayBuffer(1, 1, 10)
    buffer.store(np.zeros(1), np.ones(1), 0, np.ones(1), False)
    print(buffer.sample_batch(1))
