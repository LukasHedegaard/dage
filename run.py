from utils.parse_args import parse_args
import utils.dataset_gen as dsgen
import models
import tensorflow as tf
keras = tf.compat.v2.keras

def main(args):

    # data
    INPUT_SHAPE = (224, 224, 3)
    OUTPUT_SHAPE = len(dsgen.office31_class_names())

    ds = dsgen.office31_datasets( source_name=args.source, target_name=args.target, img_size=INPUT_SHAPE[:2], seed=args.seed)

    test_ds = [ ds['target']['test'] ]

    train_ds = {
        'tune_source': lambda: [ ds['source']['full']],
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

    test_ds  = list(map(lambda d: ( dsgen.prep_test(dataset=d['ds'], batch_size=args.batch_size) , d['size']), test_ds  ))
    train_ds = list(map(lambda d: ( dsgen.prep_train(dataset=d['ds'], batch_size=args.batch_size) , d['size']), train_ds ))

    # model
    model_base = {
        'vgg16'      : lambda: keras.applications.vgg16.VGG16      (input_shape=INPUT_SHAPE, include_top=False, weights='imagenet'),
        'resnet50'   : lambda: keras.applications.resnet50.ResNet50(input_shape=INPUT_SHAPE, include_top=False, weights='imagenet'),
    }[args.model_base]()

    model, loss = {
        'tune_source': lambda: ( models.classic(model_base, output_shape=OUTPUT_SHAPE), keras.losses.categorical_crossentropy ),
        'tune_both'  : lambda: ( models.classic(model_base, output_shape=OUTPUT_SHAPE), keras.losses.categorical_crossentropy ),
        # 'ccsa'       : lambda: models.CCSAModel(output_dim=OUTPUT_SHAPE),
        # 'dsne'       : lambda: models.DSNEModel(output_dim=OUTPUT_SHAPE),
    }[args.method]()

    optimizer = {
        'sgd' : lambda: keras.optimizers.SGD (learning_rate=args.learning_rate, momentum=0.0, nesterov=False),
        'adam': lambda: keras.optimizers.Adam(learning_rate=args.learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
    }[args.optimizer]()

    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    # perform training and test
    if 'train' in args.mode:
        for x, s in train_ds:
            model.fit( x=x, epochs=args.epochs, steps_per_epoch=s//args.batch_size, validation_split=0.0, verbose=1 )

    if 'test' in args.mode:
        for x, s in test_ds:
            model.evaluate(x, steps=s//args.batch_size, verbose=0)


if __name__ == '__main__':
    args = parse_args()
    main(args)