import argparse

def parse_args(raw_args=None):
     parser = argparse.ArgumentParser(description='Perform Supervised Domain Adaption')

     info_parser = parser.add_argument_group(description='Info')
     info_parser.add_argument('--experiment_id', type=str, default='', help='Experiment ID')
     info_parser.add_argument('--timestamp', type=str, default='', help='Timestamp')

     os_parser = parser.add_argument_group(description='OS')
     os_parser.add_argument('--gpu_id', type=str, default='2', help='Which GPU to use. Default: 2')
     os_parser.add_argument('--delete_checkpoint', type=int, default=0, help='Delete checkpoint after model training. Default: 0')
     
     # model
     model_parser = parser.add_argument_group(description='Model')
     model_parser.add_argument('--model_base', type=str, default='vgg16', 
                              help='Feature extractor for model:'
                                   'vgg16: Convolutional layers of the Visual Geometry Group model with 16 layers, pretrained on ImageNet;'
                                   'resnet101v2: Convolutional layers of the Residual Neural Network model with 101 layers, pretrained on ImageNet;'
                                   'none: dummy model containing a unity mapping;'
                              )
     model_parser.add_argument('--from_weights', type=str, default=None, help='Path to pretrained model weights')
     model_parser.add_argument('--num_unfrozen_base_layers', type=int, default=0, help='Number of unfrozen layers in base network. Default: 0')
     model_parser.add_argument('--embed_size', type=int, default=128, help='Size of embedding layer. Default: 128')
     model_parser.add_argument('--dense_size', type=int, default=1024, help='Size of first dense layer. Default: 1024')
     model_parser.add_argument('--l2', type=float, default=1e-4, help='L2 normalisation parameter. Default: 1e-4')
     model_parser.add_argument('--batch_norm', type=int, default=1, help='Use batch normalisation layers. Default: 1')
     model_parser.add_argument('--dropout', type=float, default=0.25, help='Dropout. Default: 0.25')
     model_parser.add_argument('--aux_dense_size', type=int, default=31, help='Size of aux dense layer. Default: 31')
     model_parser.add_argument('--architecture', type=str, default='single_stream',
                              help='Architectures: '
                                   'single_stream: Single stream: model_base -> Two dense layers with Relu -> classification laye;.'
                                   'two_stream_pair_embeds: Two streams: model_base -> Two dense layers with Relu -> a: aux output collecting both streams, b: -> Dense & softmax (eval for each stream);'
                                   'two_stream_pair_logits: Two streams: model_base -> Two dense layers with Relu -> Dense (no activation) -> a: aux output collecting both streams, b: softmax (eval for each stream);'
                                   'two_stream_pair_aux_dense: Two streams: model_base -> Two dense layers with Relu -> a: Dense & softmax (eval for each stream), b: Dense (no activation) -> aux output collecting both streams;'
                                   )
     # loss
     loss_parser = parser.add_argument_group(description='Loss')
     loss_parser.add_argument('--method', type=str, default='ccsa',
                              help='Methods: '
                                   'tune_source: Train one source, test on target. Should be used with single_stream architeture;'
                                   'tune_target: Train one target, test on target. Should be used with single_stream architeture;'
                                   'ccsa: Contrastive loss; Should be used with two_stream architeture'
                                   'dsne: Modified-Hausdorffian distance loss. Should be used with two_stream architeture;'
                                   'dage: Domain Adaptation using Graph Embedding. Should be used with two_stream architeture;'
                                   )
     loss_parser.add_argument('--loss_alpha', type=float, default=0.25, help='Weighting for distance-based domain adaption losses. Default: 0.25')
     loss_parser.add_argument('--loss_weights_even', type=float, default=0.5, help='ratio of source to target loss weighting. For a value of 0, only the source is weighted.')
     loss_parser.add_argument('--connection_type', type=str, default="SOURCE_TARGET", 
                              help='How the edge weights for the intrinsic graph should be connected (Only applies for method=dage):'
                                   'ALL: Inter and intra domain connections are made;'
                                   'SOURCE_TARGET: Intra domain connections are made;'
                                   'SOURCE_TARGET_PAIR: Intra domain connections are made pairwise, according to how data is fed;'
                                   'ST_INT_ALL_PEN: Inter and intra domain connections are made for the intrinsic graph, and only inter domain connection are made on the penalty graph ;' )
     loss_parser.add_argument('--weight_type', type=str, default="INDICATOR", 
                              help='How the edge weights are weighted (Only applies for method=dage):'
                                   'INDICATOR: Binary connection;'
                                   'GAUSSIAN: exp(-dist^2) where connected, 0 otherwise;' )
     loss_parser.add_argument('--connection_filter_type', type=str, default="ALL", 
                              help='How the weights are filtered (Only applies for method=dage):'
                                   'ALL: All edges are used;'
                                   'KNN: Edges for the k nearest neighbors are used;'
                                   'KFN: Edges for the k furthest neighbors are used;'
                                   'EPSILON: Edges where distances are within a distance threshold epsilon are used;' )
     loss_parser.add_argument('--penalty_connection_filter_type', type=str, default="ALL", 
                              help='How the penalty weights are filtered (Only applies for method=dage):'
                                   'ALL: All edges are used;'
                                   'KNN: Edges for the k nearest neighbors are used;'
                                   'KFN: Edges for the k furthest neighbors are used;'
                                   'EPSILON: Edges where distances are within a distance threshold epsilon are used;' )
     loss_parser.add_argument('--connection_filter_param', type=float, default=1, help='Parameter for the connection_filter_type. If loss only takes a single parameter (margin for CCSA and d-SNE), it is specified here. Default: 1')
     loss_parser.add_argument('--penalty_connection_filter_param', type=float, default=1, help='Parameter for the penalty_connection_filter_type. Default: 1')
     loss_parser.add_argument('--attention_activation', type=str, default='softmax', help='Activation function for attention. Default: softmax')

     # train
     train_parser = parser.add_argument_group(description='Training')
     train_parser.add_argument('--training_regimen', type=str, default='regular', help='How train the mode:'
                                   'regular: Train model using a regular model.fit;'
                                   'flipping: Train the model, flipping which domain enters which stream on every batch;'
                                   'gradual_unfreeze: Gradually unfreeze the base_model layers')
     train_parser.add_argument('--source', type=str, default='A', help='Source domain')
     train_parser.add_argument('--target', type=str, default='D', help='Target domain')
     train_parser.add_argument('--num_source_samples_per_class', type=int, default=8, help='Parameter only used for digits data.')
     train_parser.add_argument('--num_target_samples_per_class', type=int, default=3, help='Parameter only used for digits data.')
     train_parser.add_argument('--num_val_samples_per_class', type=int, default=3, help='Parameter only used for digits data.')
     train_parser.add_argument('--batch_size', '-b', type=int, default=16, help='Batch size. Default: 16')
     train_parser.add_argument('--epochs', '-e', type=int, default=10, help='Number of epochs. Default: 10')
     train_parser.add_argument('--seed', type=int, default=None, help='Random seed. If no argument is given, a random seed is used.')
     train_parser.add_argument('--augment', type=int, default=0, help='Activate augmentation. Only works with features="images". Default: 0')
     train_parser.add_argument('--standardize_input', type=int, default=0, help='Standardize input features. Default: 0')
     train_parser.add_argument('--resize_mode', type=int, default=0, 
                              help='How to resize images. Options: '
                                   '1: match source domain dimensions.'
                                   '2: match target domain dimensions. '
                                   '3: maximum height & width of the domains. '
                                   '4: minimum height & width of the domains. '
                              )
     train_parser.add_argument('--features', type=str, default='images', 
                              help='The input features to use. Default: images. Options: '
                                   'images: unprocessed JPG images. '
                                   'surf: speeded-up robust features.'
                                   'decaf: features extracted through DeCaf network.'
                                   'vgg16: features extracted through conv layers of a VGG16 network pretrained on ImageNet.'
                                   'resnet101v2: features extracted through conv layers of a ResNet101v2 network pretrained on ImageNet.')
     train_parser.add_argument('--mode', type=str, default='train_and_test', 
                                   help='Modes: '
                                        'train: perform training, skip evaluation;'
                                        'test: skip training, perform testing; '
                                        'train_and_validate: perform training and produce results for validation data; '
                                        'train_and_test: perform training and testing; '
                                        'train_test_validate: perform training and produce results for validation and test data; '
                                        )
     train_parser.add_argument('--ratio', type=float, default=1, help='Ratio of negative to positive pairs for domain adaption. Default: 1')
     train_parser.add_argument('--shuffle_buffer_size', type=int, default=5000, help='Size of buffer used for shuffling')
     train_parser.add_argument('--verbose', type=int, default=1, help='Verbosity of training and evaluation')
     train_parser.add_argument('--batch_repeats', type=int, default=1, help='Number of times to repeat the same batch during training. Default: 1')
     train_parser.add_argument('--test_as_val', type=int, default=0, help='Use test data for validation. Default: 0')

     # optimizer
     optim_parser = parser.add_argument_group(description='Optimization')
     optim_parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer')
     optim_parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
     optim_parser.add_argument('--learning_rate_decay', type=float, default=0.0, help='Learning rate decay')
     optim_parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
     optim_parser.add_argument('--monitor', type=str, default='loss', help='Quantity to monitor during validation: "loss" of "acc". Default: "loss"')
     
     return parser.parse_args(raw_args)
