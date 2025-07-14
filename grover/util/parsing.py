"""
The parsing functions for the argument input.
"""
import os
import pickle
from argparse import ArgumentParser, Namespace
from tempfile import TemporaryDirectory

import torch

from grover.data.molfeaturegenerator import get_available_features_generators
from grover.util.utils import makedirs


def add_common_args(parser: ArgumentParser):
    parser.add_argument('--no_cache', action='store_true', default=True,
                        help='Turn off caching mol2graph computation')
    parser.add_argument('--gpu', type=int, default=0,
                        choices=list(range(torch.cuda.device_count())),
                        help='Which GPU to use')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='Turn off cuda')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--test', type=bool, default=False,
                        help='Whether to skip training and only test the model')


def add_predict_args(parser: ArgumentParser):
    """
    Adds predict arguments to an ArgumentParser.

    :param parser: An ArgumentParser.
    """
    add_common_args(parser)

    parser.add_argument('--data_path', type=str,
                        help='Path to CSV file containing testing data for which predictions will be made')

    parser.add_argument('--output_path', type=str,
                        help='Path to CSV file where predictions will be saved')
    parser.add_argument('--checkpoint_dir', type=str,
                        help='Directory from which to load model checkpoints'
                             '(walks directory and ensembles all models that are found)')

    parser.add_argument('--features_generator', type=str, nargs='*',
                        choices=get_available_features_generators(),
                        help='Method of generating additional features')
    parser.add_argument('--features_path', type=str, nargs='*',
                        help='Path to features to use in FNN (instead of features_generator)')
    parser.add_argument('--no_features_scaling', action='store_true', default=False,
                        help='Turn off scaling of features')


def add_fingerprint_args(parser):
    add_common_args(parser)
    # parameters for fingerprints generation
    # parser.add_argument('--data_path', type=str, help='Input csv file which contains SMILES')
    # parser.add_argument('--output_path', type=str,
    #                     help='Path to model file will be saved')
    parser.add_argument('--dataset', type=str, help='dataset')
    parser.add_argument('--fold', type=int, help='fold')
    # parser.add_argument('--features_path', type=str, nargs='*',
    #                     help='Path to features to use in FNN (instead of features_generator)')
    parser.add_argument('--smiles_model_path', type=str, help='smiles model path')
    parser.add_argument('--model_path', type=str, default = None, help='model path')
    parser.add_argument('--s1', type=str, default = None, help='s1 model path')
    parser.add_argument('--s2', type=str, default = None, help='s2 model path')
    parser.add_argument('--mol_model_path', type=str, help='mol model path')
    parser.add_argument('--hid_features', type=int, default=64, help='model path')
    parser.add_argument('--dropout', type=float, default=0, help='model path')
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--device', type=str, default=None, help='device')
    parser.add_argument('--data_dir', type=str, default=None, help='datadir')



def add_finetune_args(parser: ArgumentParser):
    """
    Adds training arguments to an ArgumentParser.

    :param parser: An ArgumentParser.
    """

    # General arguments
    add_common_args(parser)
    parser.add_argument('--tensorboard', action='store_true', default=False, help='Add tensorboard logger')

    # Data argumenets
    parser.add_argument('--data_path', type=str,
                        help='Path to data CSV file.')
    parser.add_argument('--use_compound_names', action='store_true', default=False,
                        help='Use when test data file contains compound names in addition to SMILES strings')
    parser.add_argument('--max_data_size', type=int,
                        help='Maximum number of data points to load')
    # Disable this option due to some bugs.
    # parser.add_argument('--test', action='store_true', default=False,
    #                     help='Whether to skip training and only test the model')
    parser.add_argument('--features_only', action='store_true', default=False,
                        help='Use only the additional features in an FFN, no graph network')
    parser.add_argument('--features_generator', type=str, nargs='*',
                        choices=get_available_features_generators(),
                        help='Method of generating additional features.')
    parser.add_argument('--features_path', type=str, nargs='*',
                        help='Path to features to use in FNN (instead of features_generator).')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory where model checkpoints will be saved')
    parser.add_argument('--save_smiles_splits', action='store_true', default=False,
                        help='Save smiles for each train/val/test splits for prediction convenience later')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Directory from which to load model checkpoints'
                             '(walks directory and ensembles all models that are found)')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Path to model checkpoint (.pt file)')

    # Data splitting.
    parser.add_argument('--dataset_type', type=str,
                        choices=['classification', 'regression'], default='classification',
                        help='Type of dataset, e.g. classification or regression.'
                             'This determines the loss function used during training.')
    parser.add_argument('--separate_val_path', type=str,
                        help='Path to separate val set, optional')
    parser.add_argument('--separate_val_features_path', type=str, nargs='*',
                        help='Path to file with features for separate val set')
    parser.add_argument('--separate_test_path', type=str,
                        help='Path to separate test set, optional')
    parser.add_argument('--separate_test_features_path', type=str, nargs='*',
                        help='Path to file with features for separate test set')
    parser.add_argument('--split_type', type=str, default='random',
                        choices=['random', 'scaffold_balanced', 'predetermined', 'crossval', 'index_predetermined'],
                        help='Method of splitting the data into train/val/test')
    parser.add_argument('--split_sizes', type=float, nargs=3, default=[0.8, 0.1, 0.1],
                        help='Split proportions for train/validation/test sets')
    parser.add_argument('--num_folds', type=int, default=1,
                        help='Number of folds when performing cross validation')
    parser.add_argument('--folds_file', type=str, default=None,
                        help='Optional file of fold labels')
    parser.add_argument('--val_fold_index', type=int, default=None,
                        help='Which fold to use as val for leave-one-out cross val')
    parser.add_argument('--test_fold_index', type=int, default=None,
                        help='Which fold to use as test for leave-one-out cross val')
    parser.add_argument('--crossval_index_dir', type=str,
                        help='Directory in which to find cross validation index files')
    parser.add_argument('--crossval_index_file', type=str,
                        help='Indices of files to use as train/val/test'
                             'Overrides --num_folds and --seed.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed to use when splitting data into train/val/test sets.'
                             'When `num_folds` > 1, the first fold uses this seed and all'
                             'subsequent folds add 1 to the seed.')

    # Metric
    parser.add_argument('--metric', type=str, default=None,
                        choices=['auc',
                                 'prc-auc',
                                 'rmse',
                                 'mae',
                                 'r2',
                                 'accuracy',
                                 'recall',
                                 'sensitivity',
                                 'specificity',
                                 'matthews_corrcoef'],
                        help='Metric to use during evaluation.'
                             'Note: Does NOT affect loss function used during training'
                             '(loss is determined by the `dataset_type` argument).'
                             'Note: Defaults to "auc" for classification and "rmse" for regression.')
    parser.add_argument('--show_individual_scores', action='store_true', default=False,
                        help='Show all scores for individual targets, not just average, at the end')




    # Training arguments
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs to task')
    parser.add_argument('--warmup_epochs', type=float, default=2.0,
                        help='Number of epochs during which learning rate increases linearly from'
                             'init_lr to max_lr. Afterwards, learning rate decreases exponentially'
                             'from max_lr to final_lr.')
    parser.add_argument('--init_lr', type=float, default=1e-4,
                        help='Initial learning rate')
    parser.add_argument('--max_lr', type=float, default=1e-3,
                        help='Maximum learning rate')
    parser.add_argument('--final_lr', type=float, default=1e-4,
                        help='Final learning rate')
    parser.add_argument('--no_features_scaling', action='store_true', default=False,
                        help='Turn off scaling of features')
    parser.add_argument('--early_stop_epoch', type=int, default=1000, help='If val loss did not drop in '
                                                                           'this epochs, stop running')

    # Model arguments
    parser.add_argument('--ensemble_size', type=int, default=1,
                        help='Number of models for ensemble prediction.')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout probability')
    parser.add_argument('--activation', type=str, default='ReLU',
                        choices=['ReLU', 'LeakyReLU', 'PReLU', 'tanh', 'SELU', 'ELU'],
                        help='Activation function')
    parser.add_argument('--ffn_hidden_size', type=int, default=None,
                        help='Hidden dim for higher-capacity FFN (defaults to hidden_size)')
    parser.add_argument('--ffn_num_layers', type=int, default=2,
                        help='Number of layers in FFN after MPN encoding')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight_decay')
    parser.add_argument('--select_by_loss', action='store_true', default=False,
                        help='Use validation loss as refence standard to select best model to predict')

    parser.add_argument("--embedding_output_type", default="atom", choices=["atom", "bond", "both"],
                        help="This the model parameters for pretrain model. The current finetuning task only use the "
                             "embeddings from atom branch. ")

    # Self-attentive readout.
    parser.add_argument('--self_attention', action='store_true', default=False, help='Use self attention layer. '
                                                                                     'Otherwise use mean aggregation '
                                                                                     'layer.')
    parser.add_argument('--attn_hidden', type=int, default=4, nargs='?', help='Self attention layer '
                                                                              'hidden layer size.')
    parser.add_argument('--attn_out', type=int, default=128, nargs='?', help='Self attention layer '
                                                                             'output feature size.')

    parser.add_argument('--dist_coff', type=float, default=0.1, help='The dist coefficient for output of two branches.')


    parser.add_argument('--bond_drop_rate', type=float, default=0, help='Drop out bond in molecular.')
    parser.add_argument('--distinct_init', action='store_true', default=False,
                        help='Using distinct weight init for model ensemble')
    parser.add_argument('--fine_tune_coff', type=float, default=1,
                        help='Enable distinct fine tune learning rate for fc and other layer')

    # For multi-gpu finetune.
    parser.add_argument('--enbl_multi_gpu', dest='enbl_multi_gpu',
                        action='store_true', default=False,
                        help='enable multi-GPU training')

def update_checkpoint_args(args: Namespace):
    """
    Walks the checkpoint directory to find all checkpoints, updating args.checkpoint_paths and args.ensemble_size.

    :param args: Arguments.
    """
    if hasattr(args, 'checkpoint_paths') and args.checkpoint_paths is not None:
        return
    if not hasattr(args, 'checkpoint_path'):
        args.checkpoint_path = None

    if not hasattr(args, 'checkpoint_dir'):
        args.checkpoint_dir = None

    if args.checkpoint_dir is not None and args.checkpoint_path is not None:
        raise ValueError('Only one of checkpoint_dir and checkpoint_path can be specified.')

    if args.checkpoint_dir is None:
        args.checkpoint_paths = [args.checkpoint_path] if args.checkpoint_path is not None else None
        return

    args.checkpoint_paths = []

    for root, _, files in os.walk(args.checkpoint_dir):
        for fname in files:
            if fname.endswith('.pt'):
                args.checkpoint_paths.append(os.path.join(root, fname))

    if args.parser_name == "eval":
        assert args.ensemble_size * args.num_folds == len(args.checkpoint_paths)

    args.ensemble_size = len(args.checkpoint_paths)



    if args.ensemble_size == 0:
        raise ValueError(f'Failed to find any model checkpoints in directory "{args.checkpoint_dir}"')

def modify_fingerprint_args(args):
    # update_checkpoint_args(args)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    del args.no_cuda
    # makedirs(args.output_path, isfile=True)
    # setattr(args, 'fingerprint', True)


def parse_args() -> Namespace:
    """
    Parses arguments for training and testing (includes modifying/validating arguments).

    :return: A Namespace containing the parsed, modified, and validated args.
    """
    parser = ArgumentParser()
    add_fingerprint_args(parser)

    args = parser.parse_args()

    modify_fingerprint_args(args)

    return args
