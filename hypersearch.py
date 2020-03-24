import argparse
from datetime import datetime
import dill  # import needed for the checkpoint pickle to work
import pickle
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
import json
import numpy as np
import os
from pathlib import Path
from pprint import pprint
import pushover
from skopt import gp_minimize, callbacks, load
from skopt.callbacks import (
    CheckpointSaver,
    VerboseCallback,
)  # if this line below causes problems, install scikit-optimize using: pip install git+https://github.com/scikit-optimize/scikit-optimize/
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from subprocess import call
from multiprocessing import Pool


def parse_args():
    parser = argparse.ArgumentParser(
        description="Hyperparameter search using Bayesian optimization"
    )

    storage = parser.add_argument_group(description="Storage")
    storage.add_argument(
        "--id", type=str, default="", help="Identifyer for the hypersearch"
    )
    storage.add_argument(
        "--from_checkpoint",
        type=str,
        default="",
        help="Path to directory containing checkpoint to continue search from",
    )
    storage.add_argument(
        "--verbose", type=int, default=1, help="Whether to print status outputs"
    )

    search = parser.add_argument_group(description="Search")
    search.add_argument(
        "--n_random_starts", type=int, default=10, help="Number of random starts"
    )
    search.add_argument("--n_calls", type=int, default=50, help="Number of calls")
    search.add_argument(
        "--acq_func", type=str, default="EI", help="Acquisition function"
    )
    search.add_argument(
        "--noise", type=float, default=None, help="Expected noise level in optimization"
    )
    search.add_argument(
        "--seed", type=int, default=42, help="Seed for the random search"
    )

    runner = parser.add_argument_group(description="Run")
    runner.add_argument(
        "--method", type=str, default="dage", help="Domain adaptaiton method"
    )
    runner.add_argument("--source", type=str, default="D", help="Source domain")
    runner.add_argument("--target", type=str, default="W", help="Target domain")
    runner.add_argument(
        "--experiment", type=str, default="office", help="office or digits"
    )
    runner.add_argument("--gpu_id", type=str, default=None, help="gpu id")

    return parser.parse_args()


def space2dict(arr):
    return {p._name: p for p in arr}


def dict2space(dict):
    return [v for v in dict.values()]


from run import main as nn


def run(args):
    if True:  # verbose:
        print("Training objective function with parameters:")
        pprint(args)

    if args["method"] == "dage":
        method_args = [
            "--method",
            str(args["method"]),
            "--connection_type",
            "SOURCE_TARGET",
            "--weight_type",
            "INDICATOR",
            "--connection_filter_type",
            "ALL",
            "--penalty_connection_filter_type",
            "ALL",
            "--connection_filter_param",
            str(args["loss_param_1"]),
            "--penalty_connection_filter_param",
            str(args["loss_param_2"]),
        ]
    else:
        method_args = [
            "--method",
            str(args["method"]),
            "--connection_filter_param",
            str(args["loss_param_1"]),
        ]

    if args["experiment"] == "office":
        experiment_args = [
            "--from_weights",
            "./runs/tune_source/vgg16_aug_ft_best/{}{}/checkpoints/cp-best.ckpt".format(
                args["source"], args["target"]
            ),
            "--num_unfrozen_base_layers",
            str(args["num_unfrozen"]),
            "--epochs",
            "20",
            "--architecture",
            "two_stream_pair_embeds",
            "--model_base",
            "vgg16",
            "--batch_size",
            "16",
        ]
    elif args["experiment"] == "digits":
        experiment_args = [
            "--num_source_samples_per_class",
            "200",
            "--num_target_samples_per_class",
            "3",
            "--num_val_samples_per_class",
            "10",
            "--epochs",
            "50",
            "--architecture",
            "two_stream_pair_embeds",
            "--model_base",
            "conv2",
            "--dense_size",
            "120",
            "--embed_size",
            "84",
            "--batch_size",
            "128",
        ]
    else:
        raise ValueError("Unknown args.experiment: {}".format(args["experiment"]))

    acc = nn(
        [
            "--gpu_id",
            str(args["gpu_id"]),
            "--source",
            str(args["source"]),
            "--target",
            str(args["target"]),
            "--l2",
            str(args["l2"]),
            "--dropout",
            str(args["dropout"]),
            "--loss_alpha",
            str(args["alpha"]),
            "--loss_weights_even",
            str(args["ce_ratio"]),
            "--learning_rate",
            str(args["lr"]),
            "--learning_rate_decay",
            str(args["lr_decay"]),
            "--momentum",
            str(1 - args["inv_mom"]),
            "--batch_norm",
            str(args["bn"]),
            "--seed",
            str(args["seed"]),
            "--experiment_id",
            "optimizer",
            "--optimizer",
            "adam",
            "--mode",
            "train_and_test",
            "--val_as_test",
            "1",
            "--training_regimen",
            "batch_repeat",
            "--augment",
            "0",
            "--ratio",
            "3",
            "--shuffle_buffer_size",
            "3000",
            "--delete_checkpoint",
            "1",
        ]
        + method_args
        + experiment_args
    )
    return -acc


def make_obj_fun(obj_fun_args):
    @use_named_args(obj_fun_args)
    def obj_fun(**args):
        # Issue: The memory allocation of tensorflow models not bound to the tf.session life-time, but the process lifetime.
        # Workaround: Call the model creation and evaluation through another process with a scoped life-time
        # https://github.com/tensorflow/tensorflow/issues/17048
        with Pool(1) as p:
            return p.apply(run, (args,))

    return obj_fun


def make_obj_fun_dummy(obj_fun_args):
    @use_named_args(obj_fun_args)
    def obj_fun(**args):
        return -sum(args.values())

    return obj_fun


def main(args):
    run_id = args.id or datetime.now().strftime("%Y%m%d%H%M%S")
    seed = args.seed or np.random.randint(1000)
    noise = args.noise or "gaussian"
    verbose = args.verbose

    run_dir = Path(__file__).parent / "runs" / "optimizer" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # store arguments
    with open(run_dir / "config.json", "w") as f:
        json.dump(args.__dict__, f, indent=4, sort_keys=True)

    # prepare checkpoints
    checkpoint_path = run_dir / "checkpoint.pkl"
    callbacks = [
        CheckpointSaver(str(checkpoint_path.resolve()), compress=9)
    ]  # keyword arguments will be passed to `skopt.dump`
    if verbose:
        callbacks.append(
            VerboseCallback(
                n_total=args.n_calls
                + (0 if args.from_checkpoint else args.n_random_starts)
            )
        )

    # prepare search space
    experiement_search = {
        "office": [Integer(0, 16, name="num_unfrozen")],
        "digits": [],
    }[args.experiment]

    if args.method == "dage":
        method_search = [
            Integer(1, 3, name="loss_param_1"),
            Integer(8, 128, name="loss_param_2"),
        ]
    else:
        method_search = [Real(0.001, 10, name="loss_param_1")]

    # Passing arguments from main to the objective function pickling errors, so we'll just add them a single-step search dimension instead
    main_args_space = [
        Categorical([str(v)], name=k)
        for k, v in {
            "source": args.source,
            "target": args.target,
            "gpu_id": args.gpu_id,
            "method": args.method,
            "seed": args.seed,
            "experiment": args.experiment,
        }.items()
    ]

    search_space = (
        [
            Real(1e-06, 0.1, "log-uniform", name="lr"),
            Real(0.01, 0.5, "log-uniform", name="inv_mom"),
            Real(1e-07, 0.01, "log-uniform", name="lr_decay"),
            Real(0.1, 0.8, "uniform", name="dropout"),
            Real(1e-07, 0.001, "log-uniform", name="l2"),
            Real(0.01, 0.99, "uniform", name="alpha"),
            Real(0, 1, "uniform", name="ce_ratio"),
            Integer(0, 1, name="bn"),
        ]
        + method_search
        + experiement_search
        + main_args_space
    )

    if verbose:
        print("Search space:")
        pprint(space2dict(search_space))

    # prepare objective function
    obj_fun = make_obj_fun(search_space)
    # obj_fun = make_obj_fun_dummy(search_space)

    # get previous search
    if args.from_checkpoint:
        cp_path = Path(args.from_checkpoint) / "checkpoint.pkl"
        if not cp_path.is_file():
            raise ValueError(
                """    from_checkpoint should point to a .pkl file. Got: '{}'""".format(
                    args.from_checkpoint
                )
            )
        if verbose:
            print("Continuing from checkpoint: {}".format(cp_path.resolve()))
        res = load(str(cp_path.resolve()))

        # ensure that same space keys are used
        for i, dim in enumerate(res.space.dimensions):
            if not (
                dim.name == search_space[i].name and type(dim) == type(search_space[i])
            ):
                raise ValueError(
                    """
    The checkpoint search dimensions don't match the new search dimensions.
    Checkpoint dimensions:
    {}
    New dimensions:
    {}
    """.format(
                        space2dict(res.space.dimensions), space2dict(search_space)
                    )
                )

        ok_dim_inds = [
            i
            for i, x in enumerate(res.x_iters)
            if all(
                search_space[j].low <= d and d <= search_space[j].high
                for j, d in enumerate(x)
            )
        ]

        x0 = [res.x_iters[i] for i in ok_dim_inds]
        y0 = [res.func_vals[i] for i in ok_dim_inds]

        add_search_args = {"x0": x0, "y0": y0, "n_random_starts": 0}
    else:
        add_search_args = {"n_random_starts": args.n_random_starts}

    common_search_args = {
        "func": obj_fun,
        "dimensions": search_space,
        "acq_func": args.acq_func,
        "n_calls": args.n_calls,
        "noise": noise,
        "random_state": seed,
        "callback": callbacks,
    }
    search_args = {**common_search_args, **add_search_args}

    # perform search
    res = gp_minimize(**search_args)

    # print some statistics
    if verbose:
        print("x^*=")
        pprint(res.x)
        print("f(x^*)={}".format(res.fun))


if __name__ == "__main__":
    args = parse_args()
    # main(args)
    try:
        main(args)
    except Exception as e:
        print("Exception: {}".format(e))
        pushover.Client(
            user_key=os.getenv("NOTIFICATION_USER"),
            api_token=os.getenv("NOTIFICATION_TOKEN"),
        ).send_message(
            "Hypersearch exception: {}".format(e),
            title="Hyper parameter search exception",
        )
        exit(1)
    exit(0)
