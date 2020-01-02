from datetime import datetime
from pathlib import Path
import tensorflow as tf
tf.compat.v1.enable_eager_execution() 
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
keras = tf.compat.v2.keras
import models
import utils.dataset_gen as dsg
from utils.callbacks import all as callbacks
from utils.parse_args import parse_args
from utils.evaluation import evaluate
from utils.file_io import save_json, load_json
from utils.gpu import setup_gpu
from functools import partial
from timeit import default_timer as timer
import losses as losses
from shutil import rmtree
import numpy as np

def run(args):
    if args.gpu_id:
        setup_gpu(args.gpu_id, args.verbose)

    seed = args.seed or np.random.randint(1000)

    # documentation setup
    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d%H%M%S")
    outputs_dir = Path(__file__).parent / 'runs' / args.method / args.experiment_id / '{}{}_{}_{}'.format( args.source, args.target, seed, timestamp)
    checkpoints_dir = outputs_dir / 'checkpoints'
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_path= checkpoints_dir / 'cp-best.ckpt'
    tensorboard_dir = outputs_dir / 'logs'
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    config_path = outputs_dir / 'config.json'
    model_path = outputs_dir / 'model.json'
    report_path = outputs_dir / 'report.json'
    report_val_path = outputs_dir / 'report_validation.json'

    save_json(args.__dict__, config_path)

    features_config = load_json(Path('configs/features.json').absolute())

    INPUT_SHAPE = tuple(features_config[args.features]["shape"])
    CLASS_NAMES = dsg.office31_class_names()
    OUTPUT_SHAPE = len(CLASS_NAMES)

    # prepare data
    preprocess_input = {
        'vgg16'      : lambda x: keras.applications.vgg16.preprocess_input(x, mode='tf'),
        'resnet101v2': lambda x: keras.applications.resnet_v2.preprocess_input(x, mode='tf'), #NB: tf v 1.15 has a minor bug in keras_applications.resnet. Fix: change the function signature to "def preprocess_input(x, **kwargs):""
        'none'       : lambda x: x[features_config[args.features]["mat_key"]]
    }[args.model_base] or None


    ds = dsg.office31_datasets( source_name=args.source, target_name=args.target, preprocess_input=preprocess_input, shape=INPUT_SHAPE, seed=seed, features=args.features, test_as_val=args.test_as_val)

    test_ds = ds['target']['test']
    # eval_ds = ds['target']['val']

    if args.test_as_val:
        val_ds = ds['target']['test']
    else:
        val_ds = ds['target']['val']

    val_ds= {
        **{ k: lambda: val_ds                                 for k in ['tune_source', 'tune_target']},
        **{ k: lambda: dsg.da_pair_repeat_dataset(val_ds)     for k in ['ccsa', 'dsne', 'dage', 'multitask']},
        **{ k: lambda: dsg.da_pair_alt_repeat_dataset(val_ds) for k in ['dage_a']},
    }[args.method]()

    train_ds = {
        'tune_source': lambda: ds['source']['full'],
        'tune_target': lambda: ds['target']['train'],
        **{ k: lambda: dsg.da_pair_dataset(source_ds=ds['source']['train']['ds'], target_ds=ds['target']['train']['ds'], ratio=args.ratio, shuffle_buffer_size=args.shuffle_buffer_size)
            for k in ['ccsa', 'dsne', 'dage', 'multitask'] },
        **{ k: lambda: dsg.da_pair_alt_dataset(source_ds=ds['source']['train']['ds'], target_ds=ds['target']['train']['ds'], ratio=args.ratio, shuffle_buffer_size=args.shuffle_buffer_size)
            for k in ['dage_a'] },
    }[args.method]()

    test_ds  = (dsg.prep_ds(dataset=test_ds['ds'], batch_size=args.batch_size, shuffle_buffer_size=args.shuffle_buffer_size), test_ds['size'])
    # eval_ds  = (dsg.prep_ds(dataset=eval_ds['ds'], batch_size=args.batch_size, shuffle_buffer_size=args.shuffle_buffer_size), eval_ds['size'])
    val_ds   = (dsg.prep_ds(dataset=val_ds['ds'] , batch_size=args.batch_size, shuffle_buffer_size=args.shuffle_buffer_size), val_ds['size'])
    train_ds = (dsg.prep_ds_train(dataset=train_ds['ds'], batch_size=args.batch_size, shuffle_buffer_size=args.shuffle_buffer_size), train_ds['size'])

    # prepare optimizer
    optimizer = {
        'sgd'       : lambda: keras.optimizers.SGD (learning_rate=args.learning_rate, momentum=args.momentum, nesterov=True, clipvalue=10, decay=args.learning_rate_decay),
        # 'sgd_mom'   : lambda: keras.optimizers.SGD (learning_rate=args.learning_rate, momentum=0.9, nesterov=True, clipvalue=10, decay=args.learning_rate_decay),
        'adam'      : lambda: keras.optimizers.Adam(learning_rate=args.learning_rate, beta_1=args.momentum, beta_2=0.999, amsgrad=False, clipvalue=10, decay=args.learning_rate_decay),
        'rmsprop'   : lambda: keras.optimizers.RMSprop(learning_rate=args.learning_rate, clipvalue=10, decay=args.learning_rate_decay),
    }[args.optimizer]()

    # prepare model
    model_base = {
        'vgg16'      : lambda: keras.applications.vgg16.VGG16 (input_shape=INPUT_SHAPE, include_top=False, weights='imagenet'),
        'resnet101v2': lambda: keras.applications.resnet_v2.ResNet101V2(input_shape=INPUT_SHAPE, include_top=False, weights='imagenet'),
        'none'       : lambda i=keras.layers.Input(shape=INPUT_SHAPE): keras.models.Model(inputs=i, outputs=i),
    }[args.model_base]()

    aux_loss = {
        **{ k  : lambda: losses.dummy_loss for k in ['dummy','tune_source', 'tune_target', 'multitask']},
        'ccsa' : lambda : losses.contrastive_loss( margin=args.connection_filter_param) ,
        'dsne' : lambda : losses.dnse_loss( margin=args.connection_filter_param ),
        'dage' : lambda : losses.dage_loss( connection_type=args.connection_type,
                                            weight_type=args.weight_type,
                                            filter_type=args.connection_filter_type,
                                            penalty_filter_type=args.penalty_connection_filter_type,
                                            filter_param=args.connection_filter_param,
                                            penalty_filter_param=args.penalty_connection_filter_param ),
        'dage_a' : lambda : losses.dage_attention_loss( 
                                            connection_type=args.connection_type,
                                            weight_type=args.weight_type,
                                            filter_type=args.connection_filter_type,
                                            penalty_filter_type=args.penalty_connection_filter_type,
                                            filter_param=args.connection_filter_param,
                                            penalty_filter_param=args.penalty_connection_filter_param ),
    }[args.method]()

    (model, model_test) = {
        'single_stream'         : lambda : models.single_stream.model(model_base=model_base, input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE, num_unfrozen_base_layers=args.num_unfrozen_base_layers, optimizer=optimizer, dense_size=args.dense_size, embed_size=args.embed_size, l2=args.l2, dropout=args.dropout),
        'two_stream_pair_embeds': lambda : models.two_stream_pair_embeds.model(model_base=model_base, input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE, num_unfrozen_base_layers=args.num_unfrozen_base_layers, dense_size=args.dense_size, embed_size=args.embed_size, optimizer=optimizer, batch_size=args.batch_size, aux_loss=aux_loss, loss_alpha=args.loss_alpha, loss_weights_even=args.loss_weights_even, l2=args.l2, batch_norm=args.batch_norm, dropout=args.dropout),
        'two_stream_pair_logits': lambda : models.two_stream_pair_logits.model(model_base=model_base, input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE, num_unfrozen_base_layers=args.num_unfrozen_base_layers, dense_size=args.dense_size, embed_size=args.embed_size, optimizer=optimizer, batch_size=args.batch_size, aux_loss=aux_loss, loss_alpha=args.loss_alpha, loss_weights_even=args.loss_weights_even, l2=args.l2, batch_norm=args.batch_norm),
        'two_stream_aux_denses' : lambda : models.two_stream_pair_aux_dense.model(model_base=model_base, input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE, num_unfrozen_base_layers=args.num_unfrozen_base_layers, dense_size=args.dense_size, embed_size=args.embed_size, aux_dense_size=args.aux_dense_size, optimizer=optimizer, batch_size=args.batch_size, aux_loss=aux_loss, loss_alpha=args.loss_alpha, loss_weights_even=args.loss_weights_even, l2=args.l2, batch_norm=args.batch_norm),
        'two_stream_pair_embeds_attention_mid' :  
                                  lambda : models.two_stream_pair_embeds_attention_mid.model(model_base=model_base, input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE, num_unfrozen_base_layers=args.num_unfrozen_base_layers, dense_size=args.dense_size, embed_size=args.embed_size, optimizer=optimizer, batch_size=args.batch_size, aux_loss=aux_loss, loss_alpha=args.loss_alpha, loss_weights_even=args.loss_weights_even, l2=args.l2, batch_norm=args.batch_norm),
        'two_stream_pair_embeds_attention_base' :  
                                  lambda : models.two_stream_pair_embeds_attention_base.model(model_base=model_base, input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE, num_unfrozen_base_layers=args.num_unfrozen_base_layers, dense_size=args.dense_size, embed_size=args.embed_size, optimizer=optimizer, batch_size=args.batch_size, aux_loss=aux_loss, loss_alpha=args.loss_alpha, loss_weights_even=args.loss_weights_even, l2=args.l2, batch_norm=args.batch_norm),
        'two_stream_pair_embeds_attention_mid_classwise' :  
                                  lambda : models.two_stream_pair_embeds_attention_mid_classwise.model(attention_activation=args.attention_activation, model_base=model_base, input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE, num_unfrozen_base_layers=args.num_unfrozen_base_layers, dense_size=args.dense_size, embed_size=args.embed_size, optimizer=optimizer, batch_size=args.batch_size, aux_loss=aux_loss, loss_alpha=args.loss_alpha, loss_weights_even=args.loss_weights_even, l2=args.l2, batch_norm=args.batch_norm),
    }[args.architecture]()

    val_freq = 3 if args.test_as_val else 1

    train = {
        'regular':          partial(models.common.train, checkpoints_path=checkpoints_path, val_freq=val_freq),
        'flipping':         partial(models.common.train, checkpoints_path=checkpoints_path, val_freq=val_freq, flipping=True),
        # 'flipping':         partial(models.common.train_flipping, checkpoints_path=checkpoints_path),
        'batch_repeat':     partial(models.common.train, checkpoints_path=checkpoints_path, batch_repeats=args.batch_repeats),
        # 'repeat_batch':     partial(models.common.train_repeat_batch, checkpoints_path=checkpoints_path, batch_repeats=args.batch_repeats),
        'gradual_unfreeze': partial(models.common.train_gradual_unfreeze, model_base_name=args.model_base, checkpoints_path=checkpoints_path, architecture=args.architecture),
    }[args.training_regimen]

    if args.from_weights:
        weights_path = args.from_weights
        model.load_weights(str(weights_path))

    if args.verbose:
        model.summary()

    with open(model_path, 'w') as f:
        f.write(model.to_json())

    monitor = {
        **{k: 'val_' for k in ['tune_source', 'tune_target']},
        **{k: 'val_preds_' for k in ['ccsa', 'dsne', 'dage', 'dage_a', 'multitask']},
    }[args.method] + args.monitor

    fit_callbacks = callbacks(checkpoints_path, tensorboard_dir, monitor=monitor, verbose=args.verbose)

    augment = lambda x: x
    if args.augment:
        if args.features != 'images':
            raise ValueError('augment=1 is only allowed for features="images"')
        augment = {
            **{k: partial(dsg.augment, batch_size=args.batch_size) for k in ['tune_source', 'tune_target']},
            **{k: partial(dsg.augment_pair, batch_size=args.batch_size) for k in ['ccsa', 'dsne', 'dage', 'dage_a', 'multitask']},
        }[args.method]

    # perform training and test
    if 'train' in args.mode:
        x, s = train_ds
        v_x, v_s = val_ds
        start_time = timer()
        train(model=model, datasource=augment(x), datasource_size=s, val_datasource=v_x, val_datasource_size=v_s, epochs=args.epochs, batch_size=args.batch_size, callbacks=fit_callbacks, verbose=args.verbose)
        train_time = timer() - start_time
        if args.verbose:
            print("Completed training in {} seconds".format(train_time))

    result = 0

    # if 'validate' in args.mode:
    #     x, s = eval_ds
    #     res = evaluate( model=model_test, test_dataset=x, test_size=s, batch_size=args.batch_size, report_path=report_val_path, verbose=args.verbose, target_names=CLASS_NAMES )
    
    if 'test' in args.mode:
        x, s = test_ds
        result = evaluate( model=model_test, test_dataset=x, test_size=s, batch_size=args.batch_size, report_path=report_path, verbose=args.verbose, target_names=CLASS_NAMES )


    if args.delete_checkpoint:
        try:
            rmtree(str(checkpoints_dir.resolve()))
        except:
            pass


    return result

def main(raw_args=None):
    args = parse_args(raw_args)
    result = run(args)
    return result

if __name__ == '__main__':
    main()