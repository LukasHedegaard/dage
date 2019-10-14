import os
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
keras = tf.compat.v2.keras
from utils.parse_args import parse_args
import utils.dataset_gen as dsgen
from utils.evaluation import evaluate
import models
import utils.callbacks as callbacks
from datetime import datetime
from pathlib import Path
from sklearn.metrics import classification_report
import json


def setup_gpu(gpu_id):
    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.compat.v1 import InteractiveSession

    os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_id)
    if args.verbose:
        print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)


def main(args):
    if args.gpu_id:
        setup_gpu(args.gpu_id)

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

    with open(config_path, 'w') as f:
        json.dump(args.__dict__, f, indent=4)

    # data
    INPUT_SHAPE = (224, 224, 3)
    CLASS_NAMES = dsgen.office31_class_names()
    OUTPUT_SHAPE = len(CLASS_NAMES)

    preprocess_input = {
        'vgg16'      : lambda x: keras.applications.vgg16.preprocess_input(x, mode='tf'),
        'resnet50'   : lambda x: keras.applications.resnet50.preprocess_input(x, mode='tf'),
    }[args.model_base] or None

    ds = dsgen.office31_datasets( source_name=args.source, target_name=args.target, preprocess_input=preprocess_input, img_size=INPUT_SHAPE[:2], seed=args.seed)

    test_ds = [ ds['target']['test'] ]
    val_ds =  [ ds['target']['val'] ]
    train_ds = {
        'tune_source': lambda: [ ds['source']['full'] ],
        'tune_both'  : lambda: [ ds['source']['full'], ds['target']['train'] ],
        'ccsa'       : lambda: [ dsgen.da_combi_dataset(source_ds=ds['source']['train']['ds'], 
                                                        target_ds=ds['target']['train']['ds'], 
                                                        ratio=args.ratio, 
                                                        shuffle_buffer_size=args.shuffle_buffer_size) ],
        'dsne'       : lambda: [ dsgen.da_combi_dataset(source_ds=ds['source']['train']['ds'], 
                                                        target_ds=ds['target']['train']['ds'], 
                                                        ratio=args.ratio, 
                                                        shuffle_buffer_size=args.shuffle_buffer_size) ],
    }[args.method]()

    test_ds  = list(map(lambda d: ( dsgen.prep_test(dataset=d['ds'], batch_size=args.batch_size) , d['size']), test_ds ))
    val_ds   = list(map(lambda d: ( dsgen.prep_test(dataset=d['ds'], batch_size=args.batch_size) , d['size']), val_ds ))[0]
    train_ds = list(map(lambda d: ( dsgen.prep_train(dataset=d['ds'], batch_size=args.batch_size) , d['size']), train_ds ))

    # model
    model_base = {
        'vgg16'      : lambda: keras.applications.vgg16.VGG16      (input_shape=INPUT_SHAPE, include_top=False, weights='imagenet'),
        'resnet50'   : lambda: keras.applications.resnet50.ResNet50(input_shape=INPUT_SHAPE, include_top=False, weights='imagenet'),
    }[args.model_base]()

    model, loss = {
        # 'tune_source': lambda: ( models.simple(model_base, output_shape=OUTPUT_SHAPE), keras.losses.categorical_crossentropy ),
        'tune_source': lambda: ( models.classic.model(model_base, output_shape=OUTPUT_SHAPE), keras.losses.categorical_crossentropy ),
        'tune_both'  : lambda: ( models.classic.model(model_base, output_shape=OUTPUT_SHAPE), keras.losses.categorical_crossentropy ),
        # 'ccsa'       : lambda: models.CCSAModel(output_dim=OUTPUT_SHAPE),
        # 'dsne'       : lambda: models.DSNEModel(output_dim=OUTPUT_SHAPE),
    }[args.method]()

    optimizer = {
        'sgd'       : lambda: keras.optimizers.SGD (learning_rate=args.learning_rate, momentum=0.0, nesterov=False),
        'adam'      : lambda: keras.optimizers.Adam(learning_rate=args.learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False),
        'rmsprop'   : lambda: keras.optimizers.RMSprop(learning_rate=args.learning_rate),
    }[args.optimizer]()

    if args.from_weights:
        weights_path = args.from_weights
        model.load_weights(str(weights_path))

    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    if args.verbose:
        model.summary()

    with open(model_path, 'w') as f:
        f.write(model.to_json())

    fit_callbacks = callbacks.all(checkpoints_dir, tensorboard_dir, monitor='val_loss', verbose=args.verbose)
    
    train = {
        'tune_source': lambda x, s: models.classic.train(model=model, datasource=x, datasource_size=s, val_datasource=val_ds[0], val_datasource_size=val_ds[1], epochs=args.epochs, batch_size=args.batch_size, callbacks=fit_callbacks, verbose=args.verbose),
        # 'tune_both': lambda x, s: models.classic.train(model=model, datasource=x, datasource_size=s, epochs=args.epochs, batch_size=args.batch_size, callbacks=fit_callbacks, verbose=args.verbose),
        # 'ccsa'       : lambda: models.CCSAModel(output_dim=OUTPUT_SHAPE),
        # 'dsne'       : lambda: models.DSNEModel(output_dim=OUTPUT_SHAPE),
    }[args.method]


    # perform training and test
    if 'train' in args.mode:
        for x, s in train_ds:
            train(x,s)

    if 'test' in args.mode:
        for x, s in test_ds:
            # model.evaluate(x, steps=s//args.batch_size, verbose=args.verbose)
            evaluate( model=model, test_dataset=x, test_size=s, batch_size=args.batch_size, report_path=report_path, verbose=args.verbose, target_names=CLASS_NAMES )

if __name__ == '__main__':
    args = parse_args()
    main(args)