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

def main(args):
    if args.gpu_id:
        setup_gpu(args.gpu_id, args.verbose)

    # documentation setup
    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d%H%M%S")
    outputs_dir = Path(__file__).parent / 'runs' / args.method / args.experiment_id / '{}{}_{}_{}'.format( args.source, args.target, args.seed, timestamp)
    checkpoints_dir = outputs_dir / 'checkpoints'
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir = outputs_dir / 'logs'
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    config_path = outputs_dir / 'config.json'
    model_path = outputs_dir / 'model.json'
    report_path = outputs_dir / 'report.json'

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


    ds = dsg.office31_datasets( source_name=args.source, target_name=args.target, preprocess_input=preprocess_input, shape=INPUT_SHAPE, seed=args.seed, features=args.features)

    val_ds= {
        **{ k: lambda: ds['target']['val']                             for k in ['tune_source', 'tune_target']},
        **{ k: lambda: dsg.da_pair_repeat_dataset(ds['target']['val']) for k in ['ccsa', 'dsne', 'dage', 'multitask']},
        **{ k: lambda: dsg.da_pair_alt_repeat_dataset(ds['target']['val']) for k in ['dage_a']},
    }[args.method]()

    test_ds = ds['target']['test']

    train_ds = {
        'tune_source': lambda: ds['source']['full'],
        'tune_target': lambda: ds['target']['train'],
        **{ k: lambda: dsg.da_pair_dataset(source_ds=ds['source']['train']['ds'], target_ds=ds['target']['train']['ds'], ratio=args.ratio, shuffle_buffer_size=args.shuffle_buffer_size)
            for k in ['ccsa', 'dsne', 'dage', 'multitask'] },
        **{ k: lambda: dsg.da_pair_alt_dataset(source_ds=ds['source']['train']['ds'], target_ds=ds['target']['train']['ds'], ratio=args.ratio, shuffle_buffer_size=args.shuffle_buffer_size)
            for k in ['dage_a'] },
    }[args.method]()

    test_ds  = (dsg.prep_ds(dataset=test_ds['ds'], batch_size=args.batch_size, shuffle_buffer_size=args.shuffle_buffer_size), test_ds['size'])
    val_ds   = (dsg.prep_ds(dataset=val_ds['ds'] , batch_size=args.batch_size, shuffle_buffer_size=args.shuffle_buffer_size), val_ds['size'])
    train_ds = (dsg.prep_ds_train(dataset=train_ds['ds'], batch_size=args.batch_size, shuffle_buffer_size=args.shuffle_buffer_size), train_ds['size'])

    # prepare optimizer
    optimizer = {
        'sgd'       : lambda: keras.optimizers.SGD (learning_rate=args.learning_rate, momentum=0.0, nesterov=False),
        'adam'      : lambda: keras.optimizers.Adam(learning_rate=args.learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False),
        'rmsprop'   : lambda: keras.optimizers.RMSprop(learning_rate=args.learning_rate),
    }[args.optimizer]()

    # prepare model
    model_base = {
        'vgg16'      : lambda: keras.applications.vgg16.VGG16 (input_shape=INPUT_SHAPE, include_top=False, weights='imagenet'),
        'resnet101v2': lambda: keras.applications.resnet_v2.ResNet101V2(input_shape=INPUT_SHAPE, include_top=False, weights='imagenet'),
        'none'       : lambda i=keras.layers.Input(shape=INPUT_SHAPE): keras.models.Model(inputs=i, outputs=i),
    }[args.model_base]()

    aux_loss = {
        **{ k  : lambda: losses.dummy_loss for k in ['dummy','tune_source', 'tune_target', 'multitask']},
        'ccsa' : lambda : losses.contrastive_loss,
        'dsne' : lambda : losses.dnse_loss(margin=1),
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

    (model, model_test), train = {
        'single_stream'         : lambda :( models.single_stream.model(model_base=model_base, input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE, num_unfrozen_base_layers=args.num_unfrozen_base_layers, optimizer=optimizer, dense_size=args.dense_size, embed_size=args.embed_sizem, l2=args.l2),
                                            models.single_stream.train ),
        'two_stream_pair_embeds': lambda :( models.two_stream_pair_embeds.model(model_base=model_base, input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE, num_unfrozen_base_layers=args.num_unfrozen_base_layers, dense_size=args.dense_size, embed_size=args.embed_size, optimizer=optimizer, batch_size=args.batch_size, aux_loss=aux_loss, loss_alpha=args.loss_alpha, loss_weights_even=args.loss_weights_even, l2=args.l2, batch_norm=args.batch_norm),
                                            models.two_stream_pair_embeds.train ),
        'two_stream_pair_logits': lambda :( models.two_stream_pair_logits.model(model_base=model_base, input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE, num_unfrozen_base_layers=args.num_unfrozen_base_layers, dense_size=args.dense_size, embed_size=args.embed_size, optimizer=optimizer, batch_size=args.batch_size, aux_loss=aux_loss, loss_alpha=args.loss_alpha, loss_weights_even=args.loss_weights_even, l2=args.l2, batch_norm=args.batch_norm),
                                            models.two_stream_pair_logits.train ),
        'two_stream_aux_denses' : lambda :( models.two_stream_pair_aux_dense.model(model_base=model_base, input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE, num_unfrozen_base_layers=args.num_unfrozen_base_layers, dense_size=args.dense_size, embed_size=args.embed_size, aux_dense_size=args.aux_dense_size, optimizer=optimizer, batch_size=args.batch_size, aux_loss=aux_loss, loss_alpha=args.loss_alpha, loss_weights_even=args.loss_weights_even, l2=args.l2, batch_norm=args.batch_norm),
                                            models.two_stream_pair_aux_dense.train ),
        'two_stream_pair_embeds_attention_mid' :  
                                  lambda :( models.two_stream_pair_embeds_attention_mid.model(model_base=model_base, input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE, num_unfrozen_base_layers=args.num_unfrozen_base_layers, dense_size=args.dense_size, embed_size=args.embed_size, optimizer=optimizer, batch_size=args.batch_size, aux_loss=aux_loss, loss_alpha=args.loss_alpha, loss_weights_even=args.loss_weights_even, l2=args.l2, batch_norm=args.batch_norm),
                                            models.two_stream_pair_embeds_attention_mid.train ),
        'two_stream_pair_embeds_attention_base' :  
                                  lambda :( models.two_stream_pair_embeds_attention_base.model(model_base=model_base, input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE, num_unfrozen_base_layers=args.num_unfrozen_base_layers, dense_size=args.dense_size, embed_size=args.embed_size, optimizer=optimizer, batch_size=args.batch_size, aux_loss=aux_loss, loss_alpha=args.loss_alpha, loss_weights_even=args.loss_weights_even, l2=args.l2, batch_norm=args.batch_norm),
                                            models.two_stream_pair_embeds_attention_base.train ),
        'two_stream_pair_embeds_attention_mid_classwise' :  
                                  lambda :( models.two_stream_pair_embeds_attention_mid_classwise.model(attention_activation=args.attention_activation, model_base=model_base, input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE, num_unfrozen_base_layers=args.num_unfrozen_base_layers, dense_size=args.dense_size, embed_size=args.embed_size, optimizer=optimizer, batch_size=args.batch_size, aux_loss=aux_loss, loss_alpha=args.loss_alpha, loss_weights_even=args.loss_weights_even, l2=args.l2, batch_norm=args.batch_norm),
                                            models.two_stream_pair_embeds_attention_mid_classwise.train ),
    }[args.architecture]()

    if args.from_weights:
        weights_path = args.from_weights
        model.load_weights(str(weights_path))

    if args.verbose:
        model.summary()

    with open(model_path, 'w') as f:
        f.write(model.to_json())

    monitor = {
        **{k: 'val_acc' for k in ['tune_source', 'tune_target']},
        **{k: 'val_preds_acc' for k in ['ccsa', 'dsne', 'dage', 'dage_a', 'multitask']},
    }[args.method]

    fit_callbacks = callbacks(checkpoints_dir, tensorboard_dir, monitor=monitor, verbose=args.verbose)

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

    if 'test' in args.mode:
        x, s = test_ds
        evaluate( model=model_test, test_dataset=x, test_size=s, batch_size=args.batch_size, report_path=report_path, verbose=args.verbose, target_names=CLASS_NAMES )

if __name__ == '__main__':
    args = parse_args()
    main(args)