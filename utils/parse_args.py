import argparse

def parse_args():
     parser = argparse.ArgumentParser(description='Perform Supervised Domain Adaption')

     os_parser = parser.add_argument_group(description='OS')
     os_parser.add_argument('--gpu_id', type=str, default='2', help='Which GPU to use. Default: 2')
     
     # model
     model_parser = parser.add_argument_group(description='Model')
     model_parser.add_argument('--model_base', type=str, default='vgg16', help='Feature extractor for model. Default: vgg16')
     model_parser.add_argument('--from_weights', type=str, default=None, help='Path to pretrained model weights')
     model_parser.add_argument('--loss_alpha', type=float, default=0.25, help='Weighting for distance-based domain adaption losses. Default: 0.25')
     model_parser.add_argument('--loss_weights_even', type=int, default=1, help='Use even weighting for source and target losses. Default: 1')
     model_parser.add_argument('--freeze_base', type=int, default=1, help='Freeze base network. Default: 1')
     model_parser.add_argument('--embed_size', type=int, default=128, help='Size of embedding layer. Default: 128')
     model_parser.add_argument('--dense_size', type=int, default=1024, help='Size of first dense layer. Default: 1024')
     model_parser.add_argument('--aux_dense_size', type=int, default=31, help='Size of aux dense layer. Default: 31')
     
     # train
     train_parser = parser.add_argument_group(description='Training')
     train_parser.add_argument('--method', type=str, default='tune_source',
                              help='Methods: '
                                   'tune_source: train one source, test on target;'
                                   'tune_target: train one target, test on target;'
                                   'tune_both: train source and target, test on target; '
                                   'ccsa: contrastive loss; '
                                   'dsne: modified-Hausdorffian distance loss')
     train_parser.add_argument('--source', type=str, default='A', help='Source domain')
     train_parser.add_argument('--target', type=str, default='D', help='Target domain')
     train_parser.add_argument('--batch_size', '-b', type=int, default=16, help='Batch size. Default: 16')
     train_parser.add_argument('--epochs', '-e', type=int, default=10, help='Number of epochs. Default: 10')
     train_parser.add_argument('--seed', type=int, default=0, help='Random seed')
     train_parser.add_argument('--augment', type=int, default=0, help='Activate augmentation. Only works with features="images". Default: 0')
     train_parser.add_argument('--features', type=str, default='images', 
                              help='The input features to use. Default: raw. Options: '
                                   'images: unprocessed JPG images. '
                                   'surf: speeded-up robust features.'
                                   'decaf: features extracted through DeCaf network.'
                                   'vgg16: features extracted through conv layers of a VGG16 network pretrained on ImageNet.'
                                   'resnet101v2: features extracted through conv layers of a ResNet101v2 network pretrained on ImageNet.')
     train_parser.add_argument('--mode', type=str, default='train_and_test', 
                                   help='Modes: '
                                        'train: perform training, skip evaluation;'
                                        'test: skip training, perform testing; '
                                        'train_and_test: perform training and testing')
     train_parser.add_argument('--ratio', type=float, default=1, help='Ratio of negative to positive pairs for domain adaption. Default: 1')
     train_parser.add_argument('--shuffle_buffer_size', type=int, default=5000, help='Size of buffer used for shuffling')
     train_parser.add_argument('--verbose', type=int, default=1, help='Verbosity of training and evaluation')

     # optimizer
     optim_parser = parser.add_argument_group(description='Optimization')
     optim_parser.add_argument('--optimizer', type=str, default='adam', help='optimizer')
     optim_parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
     
     
     return parser.parse_args()