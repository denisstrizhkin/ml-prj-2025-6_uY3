from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from sklearn.preprocessing import Normalizer

DATA_HEADER = [
    "t",
    "V_h",
    "P",
    "R_h",
    "T",
    "V_m",
    "R_m",
    "dR/dT",
    "Triangle",
    "Iter",
]


# Softmax function
def softmax(z):
    exp_z = np.exp(z)  # for numerical stability
    return exp_z / exp_z.sum(axis=0)


class Model:
    def __init__(self, model_path):
        # Check if GPU is available
        self._device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        print(f"Using device: {self._device}")

        self._loaded_model = torch.jit.load(
            model_path, map_location=torch.device("cpu")
        )
        self._loaded_model.to(self._device)
        self._loaded_model.eval()

    def predict(self, data: pd.DataFrame) -> NDArray:
        vec = Normalizer().fit_transform(data[DATA_HEADER[1:]].values)
        tensor = torch.FloatTensor(vec)
        # print(tensor[:5])
        with torch.no_grad():
            return self._loaded_model(tensor.to(self._device)).cpu().numpy()


def get_data_dataframe(p: Path) -> pd.DataFrame:
    df = pd.read_csv(
        p,
        skiprows=1,
        decimal=",",
        sep="\t",
        header=None,
        names=DATA_HEADER,
    )
    group_key = (df.index // 25).astype(int)
    agg_operations = dict(
        [(DATA_HEADER[0], (DATA_HEADER[0], lambda x: x.iloc[-1]))]
        + [(name, (name, "mean")) for name in DATA_HEADER[1:]]
    )
    return df.groupby(group_key).agg(**agg_operations)


def get_image(time: NDArray, data: NDArray):
    probs = []
    for i in data:
        probs.append(softmax(i) * 100)
    probs = np.array(probs)

    # Plot the probabilities
    plt.figure(figsize=(6.4, 5))
    for k in range(0, probs.shape[1]):
        plt.plot(time, probs[:, k])
    plt.title("Gas type probabilities")
    plt.xlabel("Time")
    plt.ylabel("Probability")
    plt.legend(["Bensene", "H2S", "OXylene", "Toluene"])
    plt.grid(True)

    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)

    return buf.getvalue()
