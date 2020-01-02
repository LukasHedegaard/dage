import argparse
from datetime import datetime
import dill # import needed for the checkpoint pickle to work
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
import json
import numpy as np
import os
from pathlib import Path
from pprint import pprint
import pushover
from skopt import gp_minimize, callbacks, load
from skopt.callbacks import CheckpointSaver, VerboseCallback # if this line below causes problems, install scikit-optimize using: pip install git+https://github.com/scikit-optimize/scikit-optimize/   
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from subprocess import call
from multiprocessing import Pool

def parse_args():
    parser = argparse.ArgumentParser(description='Hyperparameter search using Bayesian optimization')

    storage = parser.add_argument_group(description='Storage')
    storage.add_argument('--id', type=str, default='', help='Identifyer for the hypersearch')
    storage.add_argument('--from_checkpoint', type=str, default='', help='Path to directory containing checkpoint to continue search from')
    storage.add_argument('--verbose', type=int, default=1, help='Whether to print status outputs')
    # storage.add_argument('--gpu_ids', type=str, default='', help='IDs of the GPUs to use')
    # storage.add_argument('--source', type=str, default='W', help='Source dataset')
    # storage.add_argument('--target', type=str, default='A', help='Target dataset')

    search = parser.add_argument_group(description='Search')
    search.add_argument('--n_random_starts', type=int, default=10, help='Number of random starts')
    search.add_argument('--n_calls', type=int, default=15, help='Number of calls')
    search.add_argument('--acq_func', type=str, default='EI', help='Acquisition function')
    search.add_argument('--noise', type=float, default=None, help='Expected noise level in optimization')
    search.add_argument('--seed', type=int, default=None, help='Seed for the random search')

    param = parser.add_argument_group(description='Hyper parameters')
    param.add_argument('--lr_min', type=float, default=1e-8, help='Learning rate min')
    param.add_argument('--lr_max', type=float, default=1e-3, help='Learning rate max')
    param.add_argument('--inv_mom_min', type=float, default=0.01, help='(1-param), i.e. momentum max')
    param.add_argument('--inv_mom_max', type=float, default=0.5, help='(1-param), i.e. momentum min')
    param.add_argument('--lr_decay_min', type=float, default=1e-7, help='Learning rate decay min (relative to number of epochs)')
    param.add_argument('--lr_decay_max', type=float, default=1e-2, help='Learning rate decay max (relative to number of epochs)')
    param.add_argument('--dropout_min', type=float, default=0.1, help='Dropout min')
    param.add_argument('--dropout_max', type=float, default=0.8, help='Dropout max')
    param.add_argument('--l2_min', type=float, default=1e-7, help='L2 weight decay min')
    param.add_argument('--l2_max', type=float, default=1e-3, help='L2 weight decay max')
    param.add_argument('--alpha_min', type=float, default=0.1, help='Relative weighting of domain adaptation loss vs cross entropies')
    param.add_argument('--alpha_max', type=float, default=0.99, help='Relative weighting of domain adaptation loss vs cross entropies')
    param.add_argument('--ce_ratio_min', type=float, default=0, help='Source-target cross entropy weighting ratio min')
    param.add_argument('--ce_ratio_max', type=float, default=1, help='Source-target cross entropy weighting ratio max')
    param.add_argument('--loss_param_1_min', type=int, default=1, help='Loss parameter 1 minimum value')
    param.add_argument('--loss_param_1_max', type=int, default=3, help='Loss parameter 1 maximum value')
    param.add_argument('--loss_param_2_min', type=int, default=8, help='Loss parameter 2 minimum value')
    param.add_argument('--loss_param_2_max', type=int, default=128, help='Loss parameter 2 maximum value')
    param.add_argument('--num_unfrozen_min', type=int, default=0, help='Number of unfrozen base layer minimum')
    param.add_argument('--num_unfrozen_max', type=int, default=16, help='Number of unfrozen base layer maximum')

    return parser.parse_args()


def make_obj_fun_dummy(obj_fun_args):
    @use_named_args(obj_fun_args)
    def obj_fun(**args):
        return -sum(args.values())
    return obj_fun


from run import main as nn
def run(args):
    if True: #verbose:
        print("Training objective function with parameters:")
        pprint(args)

    acc = nn([
        "--gpu_id",                 "3", #gpu_ids,
        "--source",                 "W", #source,
        "--target",                 "A", #target,
        "--from_weights",           "./runs/tune_source/vgg16_aug_ft_best/WA/checkpoints/cp-best.ckpt",
        "--num_unfrozen_base_layers", str(args['num_unfrozen']),
        # "--seed",                   str(np.random.randint(1000)),
        "--l2",                     str(args['l2']),
        "--dropout",                str(args['dropout']),
        "--loss_alpha",             str(args['alpha']),
        "--loss_weights_even",      str(args['ce_ratio']),
        "--learning_rate",          str(args['lr']),
        "--learning_rate_decay",    str(args['lr_decay']),
        "--momentum",               str(1-args['inv_mom']),
        "--experiment_id",          "optimizer", #"optimizer{}".format(run_id),
        "--batch_norm",             "1",
        "--optimizer",              "adam",
        "--batch_size",             "16",
        "--epochs",                 "30", 
        "--augment",                "0",
        "--model_base",             "vgg16",
        "--features",               "images",
        "--architecture",           "two_stream_pair_embeds",
        "--method",                 "dsne",
        # "--connection_type",                    "SOURCE_TARGET",
        # "--weight_type",                        "INDICATOR",
        # "--connection_filter_type",             "KNN",
        # "--penalty_connection_filter_type",     "KSD",
        "--connection_filter_param",            str(args['loss_param_1']),
        # "--penalty_connection_filter_param",    str(args['loss_param_2']),
        "--mode",                   "train_and_test",
        "--training_regimen",       "batch_repeat",
        "--batch_repeats",          "1", #str(args['batch_repeats']),
        "--augment",                "1",
        # "--timestamp",              "",
        # "--test_as_val",            "0",
        # "--embed_size",             "",
        # "--dense_size",             "",
        "--ratio",                  "1", #str(args['data_ratio']),
        # "--shuffle_buffer_size",    "",
        # "--verbose",                "",
        "--delete_checkpoint",      "1",
    ])
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
        Real(1e-5,                      100,                    "log-uniform",  name='loss_param_1'),
        Integer(args.num_unfrozen_min,  args.num_unfrozen_max,                  name='num_unfrozen'),
        # Integer(args.loss_param_1_min,  args.loss_param_1_max,                  name='loss_param_1'),
        # Integer(args.loss_param_2_min,  args.loss_param_2_max,                  name='loss_param_2'),
        # Integer(1,                      3,                                      name='data_ratio'),
        # Integer(1,                      3,                                      name='batch_repeats'),
        # Categorical(['all', 'knn', 'kfn', 'ksd'],                               name='con_filt_type'),
        # Categorical(['all', 'knn', 'kfn', 'ksd'],                               name='pen_con_filt_type'),
    ]

    if verbose:
        print('Search space:')
        pprint(space2dict(search_space))

    # prepare objective function
    obj_fun = make_obj_fun(search_space)
    # obj_fun = make_obj_fun_dummy(search_space)

    # get previous search
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
    if verbose:
        print("x^*=")
        pprint(res.x)
        print("f(x^*)={}".format(res.fun))

    # notify that search is done
    pushover.Client(
        user_key=os.getenv('NOTIFICATION_USER'), 
        api_token=os.getenv('NOTIFICATION_TOKEN')
    ).send_message("Finished search with ID='{}'".format(args.id), title="Hyper parameter search")


if __name__ == '__main__':
    args = parse_args()
    main(args)