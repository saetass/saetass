"""
Generate the datasets of the wind bubble surrogate and train it, one network per seed.

The test and validation sets are generated apart from the training set, with other seeds
of the Sobol sequence, and the test set is only read by pinn_wind_bubble.py. The training
takes about two hours per seed on an NVIDIA L40S, and the datasets about six minutes on
16 cores.

Usage: python pinn_train_wind_bubble.py output_directory [--seed 0] [--device cuda]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from saetass.pinn import dataset
from saetass.pinn.network import BubbleNetwork
from saetass.pinn.surrogate import WindBubbleSurrogate
from saetass.pinn.training import Tensors, TrainingOptions, train
from saetass.pinn.wind_bubble import DELTA

#: Lowest dimensionless time of the domain of the surrogate.
FIRST_TIME = 0.01
#: Number of bubbles and seed of the Sobol sequence of each dataset.
DATASETS = {"test": (128, 1001), "val": (256, 2002), "train": (8192, 3003)}


def load(path: Path) -> dict:
    """A dataset, restricted to the times of the domain of the surrogate."""
    data = dict(np.load(path))
    keep = data["log_times"] >= np.log(FIRST_TIME) - 1e-6
    data["log_times"], data["w"] = data["log_times"][keep], data["w"][:, keep]
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--steps", type=int, default=400_000)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)

    for name, (size, seed) in DATASETS.items():
        path = arguments.output / f"{name}.npz"
        if not path.exists():
            np.savez(path, **dataset.build(size, seed, max_workers=arguments.workers))

    delta = DELTA["kolmogorov"]
    training = Tensors(load(arguments.output / "train.npz"), delta, arguments.device)
    validation = Tensors(load(arguments.output / "val.npz"), delta, arguments.device)
    torch.manual_seed(arguments.seed)
    time_bounds = np.array(
        [training.log_times[0].item(), training.log_times[-1].item()]
    )
    model = BubbleNetwork(
        torch.as_tensor(dataset.DEFAULT_BOUNDS, dtype=torch.float32),
        torch.as_tensor(time_bounds, dtype=torch.float32),
    ).to(arguments.device)
    name = f"wind_bubble_{arguments.seed}"
    options = TrainingOptions(
        steps=arguments.steps,
        batch_size=16384,
        learning_rate=1e-3,
        log_every=max(arguments.steps // 20, 1),
        seed=arguments.seed,
        checkpoint=str(arguments.output / f"{name}.ckpt"),
    )
    train(model, training, validation, options)
    WindBubbleSurrogate(model, delta, arguments.device).save(
        arguments.output / f"{name}.pt"
    )
    with open(arguments.output / f"{name}.json", "w") as file:
        json.dump(options.history, file, indent=2)


if __name__ == "__main__":
    main()
