import argparse
from datetime import datetime
import dill # import needed for the checkpoint pickle to work
import json
import numpy as np
from pathlib import Path
from pprint import pprint
from skopt import gp_minimize, callbacks, load
from skopt.callbacks import CheckpointSaver, VerboseCallback # if this line below causes problems, install scikit-optimize using: pip install git+https://github.com/scikit-optimize/scikit-optimize/   
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from subprocess import call

def parse_args():
    parser = argparse.ArgumentParser(description='Hyperparameter search using Bayesian optimization')

    storage = parser.add_argument_group(description='Storage')
    storage.add_argument('--id', type=str, default='', help='Identifyer for the hypersearch')
    storage.add_argument('--from_checkpoint', type=str, default='', help='Path to directory containing checkpoint to continue search from')
    storage.add_argument('--verbose', type=int, default=1, help='Whether to print status outputs')

    search = parser.add_argument_group(description='Search')
    search.add_argument('--n_random_starts', type=int, default=10, help='Number of random starts')
    search.add_argument('--n_calls', type=int, default=15, help='Number of calls')
    search.add_argument('--acq_func', type=str, default='EI', help='Acquisition function')
    search.add_argument('--noise', type=float, default=None, help='Expected noise level in optimization')
    search.add_argument('--seed', type=int, default=None, help='Seed for the random search')
    search.add_argument('--obj_fun_path', type=int, default=None, help='Path to a script that executes the objective function')

    param = parser.add_argument_group(description='Hyper parameters')
    param.add_argument('--lr_min', type=float, default=1e-7, help='Learning rate min')
    param.add_argument('--lr_max', type=float, default=1e-2, help='Learning rate max')
    param.add_argument('--inv_mom_min', type=float, default=0.001, help='(1-param), i.e. momentum max')
    param.add_argument('--inv_mom_max', type=float, default=0.5, help='(1-param), i.e. momentum min')
    param.add_argument('--lr_decay_min', type=float, default=1e-6, help='Learning rate decay min (relative to number of epochs)')
    param.add_argument('--lr_decay_max', type=float, default=0.01, help='Learning rate decay max (relative to number of epochs)')
    param.add_argument('--dropout_min', type=float, default=0, help='Dropout min')
    param.add_argument('--dropout_max', type=float, default=0.6, help='Dropout max')
    param.add_argument('--l2_min', type=float, default=1e-6, help='L2 weight decay min')
    param.add_argument('--l2_max', type=float, default=1e-3, help='L2 weight decay max')
    param.add_argument('--alpha_min', type=float, default=0.1, help='Relative weighting of domain adaptation loss vs cross entropies')
    param.add_argument('--alpha_max', type=float, default=0.99, help='Relative weighting of domain adaptation loss vs cross entropies')
    param.add_argument('--ce_ratio_min', type=float, default=0, help='Source-target cross entropy weighting ratio min')
    param.add_argument('--ce_ratio_max', type=float, default=1, help='Source-target cross entropy weighting ratio max')
    param.add_argument('--loss_param_1_min', type=int, default=1, help='Loss parameter 1 minimum value')
    param.add_argument('--loss_param_1_max', type=int, default=3, help='Loss parameter 1 maximum value')
    param.add_argument('--loss_param_2_min', type=int, default=8, help='Loss parameter 2 minimum value')
    param.add_argument('--loss_param_2_max', type=int, default=128, help='Loss parameter 2 maximum value')
    # param.add_argument('--batch_size', type=float, default=64, help='The batch size')
    # param.add_argument('--epochs', type=float, default=30, help='The number of epochs to train')

    return parser.parse_args()


def make_obj_fun_dummy(obj_fun_args):
    @use_named_args(obj_fun_args)
    def obj_fun(**args):
        return -sum(args.values())
    return obj_fun


from run import main as train_nn
def make_obj_fun(obj_fun_args, run_id, seed):
    @use_named_args(obj_fun_args)
    def obj_fun(**args):
        macro_accuracy = train_nn([
            "--experiment_id",          "hyperopt{}".format(run_id),
            "--seed",                   str(seed),
            "--gpu_id",                 "0",
            "--source",                 "W",
            "--target",                 "D",
            "--l2",                     str(args['l2']),
            "--dropout",                str(args['dropout']),
            "--loss_alpha",             str(args['alpha']),
            "--loss_weights_even",      str(args['ce_ratio']),
            "--learning_rate",          str(args['lr']),
            "--momentum",               str(1-args['inv_mom']),
            "--batch_norm",             "1",
            "--optimizer",              "adam",
            "--batch_size",             "64",
            "--epochs",                 "30", 
            "--augment",                "0",
            "--from_weights",           "",
            "--model_base",             "none",
            "--features",               "vgg16",
            "--architecture",           "two_stream_pair_embeds",
            "--method",                 "dage",
            "--connection_type",                    "SOURCE_TARGET",
            "--weight_type",                        "INDICATOR",
            "--connection_filter_type",             "KNN",
            "--penalty_connection_filter_type",     "KSD",
            "--connection_filter_param",            str(args['loss_param_1']),
            "--penalty_connection_filter_param",    str(args['loss_param_2']),
            "--mode",                   "test",
            "--training_regimen",       "batch_repeat",
            "--batch_repeats",          "2",
            # "--timestamp", "",
            # "--test_as_val",            "0",
            # "--num_unfrozen_base_layers", "",
            # "--embed_size",           "",
            # "--dense_size",           "",
            # "--aux_dense_size",         "",
            # "--attention_activation",   "",
            # "--ratio",                  "",
            # "--shuffle_buffer_size",    "",
            # "--verbose",                "",
        ])
        return -macro_accuracy
    return obj_fun


def space2dict(arr):
        return {p._name: p for p in arr}

def dict2space(dict):
    return [v for v in dict.values()]


def main(args):
    run_id      = args.id or datetime.now().strftime("%Y%m%d%H%M%S")
    seed        = args.seed or np.random.randint(1000)
    noise       = args.noise or 'gaussian'
    verbose     = args.verbose

    run_dir = Path(__file__).parent / 'runs' / 'optimizer' / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # store arguments
    with open(run_dir/'config.json', 'w') as f:
        json.dump(args.__dict__, f, indent=4, sort_keys=True)

    # prepare checkpoints
    checkpoint_path = run_dir / 'checkpoint.pkl'
    callbacks = [CheckpointSaver(str(checkpoint_path.resolve()), compress=9)] # keyword arguments will be passed to `skopt.dump`
    if verbose:
        callbacks.append(VerboseCallback(n_total=args.n_calls + (0 if args.from_checkpoint else args.n_random_starts)))

    # prepare search space
    search_space  = [
        Real(args.lr_min,               args.lr_max,            "log-uniform",  name='lr'),
        Real(args.inv_mom_min,          args.inv_mom_max,       "log-uniform",  name='inv_mom'),
        Real(args.lr_decay_min,         args.lr_decay_max,      "log-uniform",  name='lr_decay'),
        Real(args.dropout_min,          args.dropout_max,       "uniform",      name='dropout'),
        Real(args.l2_min,               args.l2_max,            "log-uniform",  name='l2'),
        Real(args.alpha_min,            args.alpha_max,         "uniform",      name='alpha'),
        Real(args.ce_ratio_min,         args.ce_ratio_max,      "uniform",      name='ce_ratio'),
        Integer(args.loss_param_1_min,  args.loss_param_1_max,                  name='loss_param_1'),
        Integer(args.loss_param_2_min,  args.loss_param_2_max,                  name='loss_param_2'),
    ]

    if verbose:
        print('Search space:')
        pprint(space2dict(search_space))

    obj_fun = make_obj_fun(search_space, run_id=args.id, seed=seed)

    # decide if we should continue a previous search
    if args.from_checkpoint:
        cp_path = Path(args.from_checkpoint) / 'checkpoint.pkl'
        if not cp_path.is_file():
            raise ValueError("""    from_checkpoint should point to a .pkl file. Got: '{}'""".format(args.from_checkpoint))
        if verbose:
            print('Continuing from checkpoint: {}'.format(cp_path.resolve()))
        res = load(str(cp_path.resolve()))

        # ensure that same space keys are used
        for i, dim in enumerate(res.space.dimensions):
            if not( dim.name == search_space[i].name and type(dim) == type(search_space[i]) ):
                raise ValueError("""
    The checkpoint search dimensions don't match the new search dimensions.
    Checkpoint dimensions:
    {}
    New dimensions:
    {}
    """.format(space2dict(res.space.dimensions), space2dict(search_space)))

        ok_dim_inds = [ i   
                        for i, x in enumerate(res.x_iters)
                            if all( 
                                search_space[j].low <= d and d <= search_space[j].high
                                for j, d in enumerate(x)
                            )]

        x0 = [res.x_iters[i] for i in ok_dim_inds] 
        y0 = [res.func_vals[i] for i in ok_dim_inds] 

        add_search_args = { 'x0': x0,  'y0': y0,  'n_random_starts': 0 }
    else:
        add_search_args = { 'n_random_starts': args.n_random_starts }

    common_search_args = {
        'func': obj_fun, 
        'dimensions': search_space, 
        'acq_func': args.acq_func,
        'n_calls': args.n_calls,       
        'noise': noise,       
        'random_state': seed,  
        'callback': callbacks
    }
    search_args = { **common_search_args, **add_search_args }

    # perform search
    res = gp_minimize(**search_args)   

    # print some statistics
    print("x^*=")
    pprint(res.x)
    print("f(x^*)={}".format(res.fun))


if __name__ == '__main__':
    args = parse_args()
    main(args)