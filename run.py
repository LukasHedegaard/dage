from datetime import datetime
from pathlib import Path
import tensorflow as tf
# tf.compat.v1.enable_eager_execution() 
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
keras = tf.compat.v2.keras
import models
import utils.dataset_gen as dsg
from utils.callbacks import all as callbacks
from utils.parse_args import parse_args
import utils.evaluation as evaluation
from utils.io import save_json
from utils.gpu import setup_gpu
from functools import partial
from timeit import default_timer as timer

def main(args):
    if args.gpu_id:
        setup_gpu(args.gpu_id, args.verbose)

    # documentation setup
    outputs_dir = Path(__file__).parent / 'runs' / '{}_{}_{}_{}'.format( datetime.now().strftime("%Y%m%d%H%M%S"), args.source, args.target, args.method )
    checkpoints_dir = outputs_dir / 'checkpoints'
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir = outputs_dir / 'logs'
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    config_path = outputs_dir / 'config.json'
    model_path = outputs_dir / 'model.json'
    # pred_file_path = outputs_dir / 'predictions.csv'
    report_path = outputs_dir / 'report.json'

    save_json(args.__dict__, config_path)

    # prepare data
    preprocess_input = {
        'vgg16'      : lambda x: keras.applications.vgg16.preprocess_input(x, mode='tf'),
        'resnet101v2': lambda x: keras.applications.resnet_v2.preprocess_input(x, mode='tf'), #NB: tf v 1.15 has a minor bug in keras_applications.resnet. Fix: change the function signature to "def preprocess_input(x, **kwargs):""
    }[args.model_base] or None

    INPUT_SHAPE = (224, 224, 3)
    CLASS_NAMES = dsg.office31_class_names()
    OUTPUT_SHAPE = len(CLASS_NAMES)

    ds = dsg.office31_datasets( source_name=args.source, target_name=args.target, preprocess_input=preprocess_input, img_size=INPUT_SHAPE[:2], seed=args.seed)

    val_ds, test_ds = {
        'tune_source': lambda: ( ds['target']['val'], ds['target']['test'] ),
        'tune_target': lambda: ( ds['target']['val'], ds['target']['test'] ),
        'tune_both'  : lambda: ( ds['target']['val'], ds['target']['test'] ),
        'ccsa'       : lambda: ( dsg.da_pair_repeat_dataset(ds['target']['val']), dsg.da_pair_repeat_dataset(ds['target']['test']) ),
        'dsne'       : lambda: ( dsg.da_pair_repeat_dataset(ds['target']['val']), dsg.da_pair_repeat_dataset(ds['target']['test']) ),      
        'homebrew'   : lambda: ( dsg.da_pair_repeat_dataset(ds['target']['val']), dsg.da_pair_repeat_dataset(ds['target']['test']) ),      
    }[args.method]()

    train_ds = {
        'tune_source': lambda: ds['source']['full'],
        'tune_target': lambda: ds['target']['train'],
        'ccsa'       : lambda: dsg.da_pair_dataset(source_ds=ds['source']['train']['ds'], 
                                                   target_ds=ds['target']['train']['ds'], 
                                                   ratio=args.ratio, 
                                                   shuffle_buffer_size=args.shuffle_buffer_size),
        'dsne'       : lambda: dsg.da_pair_dataset(source_ds=ds['source']['train']['ds'], 
                                                   target_ds=ds['target']['train']['ds'], 
                                                   ratio=args.ratio, 
                                                   shuffle_buffer_size=args.shuffle_buffer_size),
        'homebrew'   : lambda: dsg.da_pair_dataset(source_ds=ds['source']['train']['ds'], 
                                                   target_ds=ds['target']['train']['ds'], 
                                                   ratio=args.ratio, 
                                                   shuffle_buffer_size=args.shuffle_buffer_size),
    }[args.method]()

    test_ds  = (dsg.prep_ds(dataset=test_ds['ds'], batch_size=args.batch_size), test_ds['size'])
    val_ds   = (dsg.prep_ds(dataset=val_ds['ds'] , batch_size=args.batch_size), val_ds['size'])
    train_ds = (dsg.prep_ds_train(dataset=train_ds['ds'], batch_size=args.batch_size), train_ds['size'])

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
    }[args.model_base]()
    model_base.summary()

    model, train = {
        'tune_source': lambda: (  models.classic.model(model_base, input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE, freeze_base=args.freeze_base, optimizer=optimizer, dense_size=args.dense_size, embed_size=args.embed_size), models.classic.train),
        'tune_target': lambda: (  models.classic.model(model_base, input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE, freeze_base=args.freeze_base, optimizer=optimizer, dense_size=args.dense_size, embed_size=args.embed_size), models.classic.train),
        'tune_both'  : lambda: (  models.classic.model(model_base, input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE, freeze_base=args.freeze_base, optimizer=optimizer, dense_size=args.dense_size, embed_size=args.embed_size), models.classic.train),
        'ccsa'       : lambda: (     models.ccsa.model(model_base, input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE, freeze_base=args.freeze_base, optimizer=optimizer, dense_size=args.dense_size, embed_size=args.embed_size, alpha=args.alpha, even_loss_weights=args.even_loss_weights), models.ccsa.train),
        'dsne'       : lambda: (     models.dsne.model(model_base, input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE, freeze_base=args.freeze_base, optimizer=optimizer, dense_size=args.dense_size, embed_size=args.embed_size, alpha=args.alpha, even_loss_weights=args.even_loss_weights, batch_size=args.batch_size), models.dsne.train),
        'homebrew'   : lambda: ( models.homebrew.model(model_base, input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE, freeze_base=args.freeze_base, optimizer=optimizer, dense_size=args.dense_size, embed_size=args.embed_size, alpha=args.alpha, even_loss_weights=args.even_loss_weights, batch_size=args.batch_size), models.homebrew.train),
    }[args.method]()

    evaluate = {
        'tune_source': evaluation.evaluate,
        'tune_target': evaluation.evaluate,
        'tune_both'  : evaluation.evaluate,
        'ccsa'       : evaluation.evaluate_da_pair,
        'dsne'       : evaluation.evaluate_da_pair,
        'homebrew'   : evaluation.evaluate_da_pair,
    }[args.method]

    if args.from_weights:
        weights_path = args.from_weights
        model.load_weights(str(weights_path))

    # model.compile(loss=loss, loss_weights=loss_weights, optimizer=optimizer, metrics=['accuracy'])

    if args.verbose:
        model.summary()

    with open(model_path, 'w') as f:
        f.write(model.to_json())

    monitor = {
        'tune_source': 'val_acc',
        'tune_target': 'val_acc',
        'tune_both'  : 'val_acc',
        'ccsa'       : 'val_preds_acc',
        'dsne'       : 'val_preds_acc',
        'homebrew'   : 'val_preds_acc',
    }[args.method]
    fit_callbacks = callbacks(checkpoints_dir, tensorboard_dir, monitor=monitor, verbose=args.verbose)

    augment = lambda x: x
    if args.augment:
        augment = {
            'tune_source': partial(dsg.augment,      batch_size=args.batch_size),
            'tune_target': partial(dsg.augment,      batch_size=args.batch_size),
            'tune_both'  : partial(dsg.augment,      batch_size=args.batch_size),
            'ccsa'       : partial(dsg.augment_pair, batch_size=args.batch_size),
            'dsne'       : partial(dsg.augment_pair, batch_size=args.batch_size),
            'homebrew'   : partial(dsg.augment_pair, batch_size=args.batch_size),
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
        evaluate( model=model, test_dataset=x, test_size=s, batch_size=args.batch_size, report_path=report_path, verbose=args.verbose, target_names=CLASS_NAMES )

if __name__ == '__main__':
    args = parse_args()
    main(args)