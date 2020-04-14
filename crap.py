import crepe
import os
import numpy as np


# TODO integrate this with data preparation pipeline
array_dir = "/cache/EnglishRaw2"

for arr in os.listdir(array_dir):
    audio = np.load(os.path.join(array_dir, arr))
    _, f0, _, _ = crepe.predict(audio[0], 16000, step_size=20, viterbi=True)
    np.save(os.path.join("/cache/EnglishF0", arr), f0)
