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

Dataset = tf.compat.v2.data.Dataset


def run(args):
    if args.gpu_id:
        setup_gpu(args.gpu_id, args.verbose)

    seed = args.seed or np.random.randint(1000)

    # documentation setup
    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d%H%M%S")
    outputs_dir = (
        Path(__file__).parent
        / "runs"
        / args.method
        / args.experiment_id
        / "{}{}_{}_{}".format(args.source, args.target, seed, timestamp)
    )
    checkpoints_dir = outputs_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_path = checkpoints_dir / "cp-best.ckpt"
    tensorboard_dir = outputs_dir / "logs"
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    config_path = outputs_dir / "config.json"
    model_path = outputs_dir / "model.json"
    report_path = outputs_dir / "report.json"
    report_val_path = outputs_dir / "report_validation.json"

    save_json(args.__dict__, config_path)

    features_config = load_json(Path("configs/features.json").absolute())

    # prepare data
    preprocess_input = {
        "vgg16": lambda x: keras.applications.vgg16.preprocess_input(x, mode="tf"),
        "resnet101v2": lambda x: keras.applications.resnet_v2.preprocess_input(
            x, mode="tf"
        ),  # NB: tf v 1.15 has a minor bug in keras_applications.resnet. Fix: change the function signature to "def preprocess_input(x, **kwargs):""
        "none": lambda x: x[features_config[args.features]["mat_key"]],
        **{k: lambda x: x for k in ["conv2", "lenetplus"]},
    }[args.model_base] or None

    if all([name in dsg.OFFICE_DATASET_NAMES for name in [args.source, args.target]]):
        # office data
        INPUT_SHAPE = tuple(features_config[args.features]["shape"])
        CLASS_NAMES = dsg.office31_class_names()
        OUTPUT_SHAPE = len(CLASS_NAMES)
        ds = dsg.office31_datasets_new(
            source_name=args.source,
            target_name=args.target,
            preprocess_input=preprocess_input,
            shape=INPUT_SHAPE,
            seed=seed,
        )

    elif all([name in dsg.DIGIT_DATASET_NAMES for name in [args.source, args.target]]):
        INPUT_SHAPE = dsg.digits_shape(args.source, args.target, mode=args.resize_mode)
        CLASS_NAMES = dsg.digits_class_names()
        OUTPUT_SHAPE = len(CLASS_NAMES)
        ds = dsg.digits_datasets_new(
            source_name=args.source,
            target_name=args.target,
            num_source_samples_per_class=args.num_source_samples_per_class,
            num_target_samples_per_class=args.num_target_samples_per_class,
            num_val_samples_per_class=args.num_val_samples_per_class,
            seed=seed,
            input_shape=INPUT_SHAPE,
            standardize_input=args.standardize_input,
        )

    else:
        raise Exception(
            "The source and target datasets should come from either Office31 or Digits"
        )

    source_all_ds, source_all_size = ds["source"]["full"]
    source_train_ds, source_train_size = ds["source"]["train"]
    target_train_ds, target_train_size = ds["target"]["train"]
    target_val_ds, target_val_size = ds["target"]["val"]
    target_test_ds, target_test_size = ds["target"]["test"]
    test_size = target_test_size

    if args.test_as_val:
        target_val_ds = target_test_ds
        target_val_size = target_test_size

    if args.val_as_test:
        target_test_ds = target_val_ds
        target_test_size = target_val_size

    val_ds, val_size = {
        **{
            k: lambda: (target_val_ds, target_val_size)
            for k in ["tune_source", "tune_target"]
        },
        **{
            k: lambda: dsg.da_pair_repeat_dataset(target_val_ds, target_val_size)
            for k in ["ccsa", "dsne", "dage", "multitask"]
        },
        **{
            k: lambda: dsg.da_pair_alt_repeat_dataset(target_val_ds, target_val_size)
            for k in ["dage_a"]
        },
    }[args.method]()

    train_ds, train_size = {
        "tune_source": lambda: (source_all_ds, source_all_size),
        "tune_target": lambda: (target_train_ds, target_train_size),
        **{
            k: lambda: dsg.da_pair_dataset(
                source_ds=source_train_ds,
                target_ds=target_train_ds,
                num_source_samples_per_class=(
                    args.num_source_samples_per_class
                    or (20 if args.source.lower()[0] == "a" else 8)
                ),
                num_target_samples_per_class=(args.num_target_samples_per_class or 3),
                num_classes=OUTPUT_SHAPE,
                ratio=args.ratio,
                shuffle_buffer_size=args.shuffle_buffer_size,
            )
            for k in ["ccsa", "dsne", "dage", "multitask"]
        },
        **{
            k: lambda: dsg.da_pair_alt_dataset(
                source_ds=source_train_ds,
                target_ds=target_train_ds,
                ratio=args.ratio,
                shuffle_buffer_size=args.shuffle_buffer_size,
            )
            for k in ["dage_a"]
        },
    }[args.method]()

    # prep data
    test_ds = dsg.prep_ds(
        dataset=target_test_ds,
        batch_size=args.batch_size,
        shuffle_buffer_size=args.shuffle_buffer_size,
    )
    val_ds = dsg.prep_ds(
        dataset=val_ds,
        batch_size=args.batch_size,
        shuffle_buffer_size=args.shuffle_buffer_size,
    )
    train_ds = dsg.prep_ds_train(
        dataset=train_ds,
        batch_size=args.batch_size,
        shuffle_buffer_size=args.shuffle_buffer_size,
    )

    # prepare optimizer
    optimizer = {
        "sgd": lambda: keras.optimizers.SGD(
            learning_rate=args.learning_rate,
            momentum=args.momentum,
            nesterov=True,
            clipvalue=10,
            decay=args.learning_rate_decay,
        ),
        "adam": lambda: keras.optimizers.Adam(
            learning_rate=args.learning_rate,
            beta_1=args.momentum,
            beta_2=0.999,
            amsgrad=False,
            clipvalue=10,
            decay=args.learning_rate_decay,
        ),
        "rmsprop": lambda: keras.optimizers.RMSprop(
            learning_rate=args.learning_rate,
            clipvalue=10,
            decay=args.learning_rate_decay,
        ),
    }[args.optimizer]()

    # prepare model
    model_base = {
        "vgg16": lambda: keras.applications.vgg16.VGG16(
            input_shape=INPUT_SHAPE, include_top=False, weights="imagenet"
        ),
        "resnet101v2": lambda: keras.applications.resnet_v2.ResNet101V2(
            input_shape=INPUT_SHAPE, include_top=False, weights="imagenet"
        ),
        "conv2": lambda: models.common.conv2_block(
            input_shape=INPUT_SHAPE,
            l2=args.l2,
            dropout=args.dropout / 2,
            batch_norm=args.batch_norm,
        ),
        "lenetplus": lambda: models.common.lenetplus_conv_block(
            input_shape=INPUT_SHAPE,
            l2=args.l2,
            dropout=args.dropout / 2,
            batch_norm=args.batch_norm,
        ),
        "none": lambda i=keras.layers.Input(shape=INPUT_SHAPE): keras.models.Model(
            inputs=i, outputs=i
        ),
    }[args.model_base]()

    aux_loss = {
        **{
            k: lambda: losses.dummy_loss
            for k in ["dummy", "tune_source", "tune_target", "multitask"]
        },
        "ccsa": lambda: losses.contrastive_loss(margin=args.connection_filter_param),
        "dsne": lambda: losses.dnse_loss(margin=args.connection_filter_param),
        "dage": lambda: losses.dage_loss(
            connection_type=args.connection_type,
            weight_type=args.weight_type,
            filter_type=args.connection_filter_type,
            penalty_filter_type=args.penalty_connection_filter_type,
            filter_param=args.connection_filter_param,
            penalty_filter_param=args.penalty_connection_filter_param,
        ),
    }[args.method]()

    (model, model_test) = {
        "single_stream": lambda: models.single_stream.model(
            model_base=model_base,
            input_shape=INPUT_SHAPE,
            output_shape=OUTPUT_SHAPE,
            num_unfrozen_base_layers=args.num_unfrozen_base_layers,
            optimizer=optimizer,
            dense_size=args.dense_size,
            embed_size=args.embed_size,
            l2=args.l2,
            dropout=args.dropout,
        ),
        "two_stream_pair_embeds": lambda: models.two_stream_pair_embeds.model(
            model_base=model_base,
            input_shape=INPUT_SHAPE,
            output_shape=OUTPUT_SHAPE,
            num_unfrozen_base_layers=args.num_unfrozen_base_layers,
            dense_size=args.dense_size,
            embed_size=args.embed_size,
            optimizer=optimizer,
            batch_size=args.batch_size,
            aux_loss=aux_loss,
            loss_alpha=args.loss_alpha,
            loss_weights_even=args.loss_weights_even,
            l2=args.l2,
            batch_norm=args.batch_norm,
            dropout=args.dropout,
        ),
    }[args.architecture]()

    val_freq = 3 if args.test_as_val else 1

    train = {
        "regular": partial(
            models.common.train, checkpoints_path=checkpoints_path, val_freq=val_freq
        ),
        "flipping": partial(
            models.common.train,
            checkpoints_path=checkpoints_path,
            val_freq=val_freq,
            flipping=True,
        ),
        "batch_repeat": partial(
            models.common.train,
            checkpoints_path=checkpoints_path,
            batch_repeats=args.batch_repeats,
        ),
        "gradual_unfreeze": partial(
            models.common.train_gradual_unfreeze,
            model_base_name=args.model_base,
            checkpoints_path=checkpoints_path,
            architecture=args.architecture,
        ),
    }[args.training_regimen]

    if args.from_weights:
        weights_path = args.from_weights
        model.load_weights(str(weights_path))

    if args.verbose:
        model.summary()
        # keras.utils.plot_model( #type: ignore
        #     model,
        #     to_file=(Path(__file__).parent /'model.png').absolute(),
        #     show_shapes=True,
        #     show_layer_names=True,
        #     rankdir='TB',
        #     expand_nested=True,
        #     dpi=96
        # )

    with open(model_path, "w") as f:
        f.write(model.to_json())

    monitor = {
        **{k: "val_" for k in ["tune_source", "tune_target"]},
        **{k: "val_preds_" for k in ["ccsa", "dsne", "dage", "dage_a", "multitask"]},
    }[args.method] + args.monitor

    fit_callbacks = callbacks(
        checkpoints_path, tensorboard_dir, monitor=monitor, verbose=args.verbose
    )

    augment = lambda x: x
    if args.augment:
        if args.features != "images":
            raise ValueError('augment=1 is only allowed for features="images"')
        augment = {
            **{
                k: partial(
                    dsg.augment, batch_size=args.batch_size, input_shape=INPUT_SHAPE
                )
                for k in ["tune_source", "tune_target"]
            },
            **{
                k: partial(
                    dsg.augment_pair,
                    batch_size=args.batch_size,
                    input_shape=INPUT_SHAPE,
                )
                for k in ["ccsa", "dsne", "dage", "dage_a", "multitask"]
            },
        }[args.method]

    # perform training and test
    if "train" in args.mode:
        start_time = timer()
        train(
            model=model,
            datasource=augment(train_ds),
            datasource_size=train_size,
            val_datasource=val_ds,
            val_datasource_size=val_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=fit_callbacks,
            verbose=args.verbose,
        )
        train_time = timer() - start_time
        if args.verbose:
            print("Completed training in {} seconds".format(train_time))

    result = 0

    if "test" in args.mode:
        result = evaluate(
            model=model_test,
            test_dataset=test_ds,
            test_size=test_size,
            batch_size=args.batch_size,
            report_path=report_path,
            verbose=args.verbose,
            target_names=CLASS_NAMES,
        )

    if args.delete_checkpoint:
        try:
            rmtree(str(checkpoints_dir.resolve()))
        except:
            pass

    return result["accuracy"]  # type:ignore


def main(raw_args=None):
    args = parse_args(raw_args)
    result = run(args)
    return result


if __name__ == "__main__":
    main()
