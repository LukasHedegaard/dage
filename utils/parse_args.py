import argparse

def parse_args():
     parser = argparse.ArgumentParser(description='Perform Supervised Domain Adaption')

     os_parser = parser.add_argument_group(description='OS')
     os_parser.add_argument('--gpu_id', type=str, default='0', help='Which GPU to use. Default: 0')
     
     # model
     model_parser = parser.add_argument_group(description='Model')
     model_parser.add_argument('--model_base', type=str, default='vgg16', help='Feature extractor for model. Default: vgg16')
     model_parser.add_argument('--from_weights', type=str, default=None, help='Path to pretrained model weights')
     
     # train
     train_parser = parser.add_argument_group(description='Training')
     train_parser.add_argument('--method', type=str, default='tune_source',
                              help='Methods: '
                                   'tune_source: train one source, test on target;'
                                   'tune_both: train source and target, test on target; '
                                   'ccsa: contrastive loss; '
                                   'dsne: modified-Hausdorffian distance loss')
     train_parser.add_argument('--source', type=str, default='A', help='Source domain')
     train_parser.add_argument('--target', type=str, default='D', help='Target domain')
     train_parser.add_argument('--batch-size', '-b', type=int, default=8, help='Batch size. Default: 8')
     train_parser.add_argument('--epochs', '-e', type=int, default=10, help='Number of epochs. Default: 10')
     train_parser.add_argument('--seed', type=int, default=0, help='Random seed')
     train_parser.add_argument('--mode', type=str, default='train_and_test', 
                                   help='Modes: '
                                        'train: perform training, skip evaluation;'
                                        'test: skip training, perform testing; '
                                        'train_and_test: perform training and testing')
     train_parser.add_argument('--ratio', type=float, default=None, help='Ratio of negative to positive pairs for domain adaption')
     train_parser.add_argument('--shuffle_buffer_size', type=int, default=1000, help='Size of buffer used for shuffling')
     train_parser.add_argument('--verbose', type=int, default=1, help='Verbosity of training and evaluation')

     # optimizer
     optim_parser = parser.add_argument_group(description='Optimization')
     optim_parser.add_argument('--optimizer', type=str, default='adam', help='optimizer')
     optim_parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
     
     
     return parser.parse_args()