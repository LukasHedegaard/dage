from utils.parse_args import parse_args
import utils.dataset_gen as dsgen
from models import ModelTrainer, ClassicModel, CCSAModel, DSNEModel

def main(args):

    n_classes = len(dsgen.office31_class_names())

    # model
    if args.method == 'tune_source':
        model = ClassicModel(output_dim=n_classes)
    elif args.method == 'tune_both':
        model = ClassicModel(output_dim=n_classes)
    elif args.method == 'ccsa':
        model = CCSAModel(output_dim=n_classes)
    elif args.method == 'dsne':
        model = DSNEModel(output_dim=n_classes)
    else:
        raise NotImplementedError

    # data
    ds = dsgen.office31_datasets(
        source_name=args.source,
        target_name=args.target,
        img_size=model.input_dim,
        seed=args.seed,
    )
    if args.method == 'tune_source':
        train_ds = [
            ds['source']['full']
        ]
    elif args.method == 'tune_both':
        train_ds = [
            ds['source']['full'],
            ds['target']['train']
        ]
    elif args.method in ['ccsa', 'dsne']:
        train_ds = [ 
            dsgen.da_combi_dataset(
                ds['source']['train'], 
                ds['target']['train'], 
                ratio=args.ratio,
                shuffle_buffer_size=args.shuffle_buffer_size
            )
        ]
    test_ds = [ds['target']['test']]

    # model trainer
    model_trainer = ModelTrainer(
        model=model,
        learning_rate=args.learning_rate,
        output_dir=''
    )

    # perform training and test
    if 'train' in args.mode:
        for d in train_ds:
            model_trainer.train(
                dsgen.prep_for_training( 
                    dataset=d, 
                    batch_size=args.batch_size,
                    shuffle_buffer_size=args.shuffle_buffer_size
                )
            )

    if 'test' in args.mode:
        for d in test_ds:
            model_trainer.test(
                dsgen.prep_for_test( 
                    dataset=d, 
                    batch_size=args.batch_size,
                )
            )


if __name__ == '__main__':
    args = parse_args()
    main(args)