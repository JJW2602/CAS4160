from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboardX import SummaryWriter
import os
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.ticker import FuncFormatter
from typing import List, Dict, Any
import numpy as np





def extract_tensorboard_scalars(log_file, scalar_keys):
    # Initialize an EventAccumulator with the path to the log directory
    event_acc = EventAccumulator(log_file)
    event_acc.Reload()  # Load the events from disk

    if isinstance(scalar_keys, str):
        scalar_keys = [scalar_keys]

    # Extract the scalar summaries
    scalars = {}
    for tag in scalar_keys:
        events = event_acc.Scalars(tag)
        scalars[tag] = {
            "step":      [e.step      for e in events],
            "wall_time": [e.wall_time for e in events],
            "value":     [e.value     for e in events],
        }

    return scalars


def compute_mean_std(scalars: List[Dict[str, Any]], data_key: str, ninterp=100):
    min_step = min([s for slog in scalars for s in slog[data_key]["step"]])
    max_step = max([s for slog in scalars for s in slog[data_key]["step"]])
    steps = np.linspace(min_step, max_step, ninterp)
    scalars_interp = np.stack(
        [
            np.interp(
                steps,
                slog[data_key]["step"],
                slog[data_key]["value"],
                left=float("nan"),
                right=float("nan"),
            )
            for slog in scalars
        ],
        axis=1,
    )

    mean = np.mean(scalars_interp, axis=1)
    std = np.std(scalars_interp, axis=1)

    return steps, mean, std


def plot_mean_std(
    ax: plt.Axes,
    steps: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    name: str,
    color: str,
):
    ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.3)
    ax.plot(steps, mean, color=color, label=name)


def plot_scalars(
    ax: plt.Axes, scalars: Dict[str, Any], data_key: str, name: str, color: str
):
    ax.plot(
        scalars[data_key]["step"], scalars[data_key]["value"], color=color, label=name
    )


if __name__ == "__main__":
    import argparse

    # Example usage
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_log_files", "-i", nargs="+", required=True)
    parser.add_argument(
        "--human_readable_names", "-n", nargs="+", default=None, required=False
    )
    parser.add_argument("--colors", "-c", nargs="+", default=None, required=False)
    parser.add_argument("--data_key", "-d", nargs="+", type=str, required=True)
    parser.add_argument("--plot_mean_std", "-std", action="store_true")
    parser.add_argument("--title", "-t", type=str, default=None, required=False)
    parser.add_argument("--x_label_name", "-x", type=str, default=None, required=False)
    parser.add_argument("--y_label_name", "-y", type=str, default=None, required=False)
    parser.add_argument("--baseline", "-b", type=int, default=None, required=False)
    parser.add_argument(
        "--output_file", "-o", type=str, default="output_plot.png", required=False
    )

    args = parser.parse_args()

    has_names = True

    if args.plot_mean_std:
        args.data_key = args.data_key[0]
        if args.colors is None:
            args.colors = [None]

        if args.human_readable_names is None:
            has_names = False
            args.human_readable_names = [None]

        assert len(args.human_readable_names) == 1
        assert len(args.colors) == 1

        all_scalars = [
            extract_tensorboard_scalars(log, args.data_key)
            for log in args.input_log_files
        ]
        xs, mean, std = compute_mean_std(all_scalars, args.data_key)
        plot_mean_std(
            plt.gca(), xs, mean, std, args.human_readable_names[0], args.colors[0]
        )
    else:
        if args.colors is None:
            args.colors = [None] * len(args.input_log_files)

        if args.human_readable_names is None:
            has_names = False
            args.human_readable_names = [None] * len(args.input_log_files)

        assert len(args.human_readable_names) == len(args.input_log_files)
        assert len(args.colors) == len(args.input_log_files)

        for log, name, color in zip(
            args.input_log_files, args.human_readable_names, args.colors
        ):
            for data_key in args.data_key:
                scalars = extract_tensorboard_scalars(log, data_key)
                if not args.plot_mean_std:
                    plot_name = f"{name}-{data_key}" if len(args.data_key) > 1 else name
                    plot_scalars(plt.gca(), scalars, data_key, plot_name, color)
    if args.baseline:
        plt.axhline(y=args.baseline, color='gray', linestyle='--', linewidth=1)
        yticks = plt.yticks()[0]
        new_yticks = list(yticks) + [args.baseline]
        plt.yticks(sorted(set(new_yticks)))

    if has_names:
        plt.legend()

    if args.title:
        plt.title(args.title)

    if args.x_label_name:
        plt.xlabel(args.x_label_name)

    if args.y_label_name:
        plt.ylabel(args.y_label_name)

    ax = plt.gca()
    ax.set_xlim(0, 300_000)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(50_000))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x/1000)}K"))
    
    plt.savefig(args.output_file)
    plt.show()
