import csv
from logging import Logger
import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import random
from argparse import Namespace
from typing import Callable, List, Union
import numpy as np
from rdkit import Chem
from tqdm import tqdm as core_tqdm
from grover.data.molfeaturegenerator import get_features_generator
from grover.data.scaler import StandardScaler

class tqdm(core_tqdm):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("ascii", True)
        super(tqdm, self).__init__(*args, **kwargs)

class MoleculeDatapoint:
    """A MoleculeDatapoint contains a single molecule and its associated features and targets."""

    def __init__(self,
                 line: List[str],
                 args: Namespace = None,
                 features: np.ndarray = None,
                 use_compound_names: bool = False):
        """
        Initializes a MoleculeDatapoint, which contains a single molecule.

        :param line: A list of strings generated by separating a line in a data CSV file by comma.
        :param args: Arguments.
        :param features: A numpy array containing additional features (ex. Morgan fingerprint).
        :param use_compound_names: Whether the data CSV includes the compound name on each line.
        """
        self.features_generator = None
        self.args = None
        if args is not None:
            if hasattr(args, "features_generator"):
                self.features_generator = args.features_generator
            self.args = args

        if features is not None and self.features_generator is not None:
            raise ValueError('Currently cannot provide both loaded features and a features generator.')

        self.features = features

        if use_compound_names:
            self.compound_name = line[0]  # str
            line = line[1:]
        else:
            self.compound_name = None

        self.smiles = line[0]  # str

        # Generate additional features if given a generator
        if self.features_generator is not None:
            self.features = []
            mol = Chem.MolFromSmiles(self.smiles)
            for fg in self.features_generator:
                features_generator = get_features_generator(fg)
                if mol is not None and mol.GetNumHeavyAtoms() > 0:
                    if fg in ['morgan', 'morgan_count']:
                        self.features.extend(features_generator(mol, num_bits=args.num_bits))
                    else:
                        self.features.extend(features_generator(mol))

            self.features = np.array(self.features)

        # Fix nans in features
        if self.features is not None:
            replace_token = 0
            self.features = np.where(np.isnan(self.features), replace_token, self.features)

        # Create targets
        self.targets = [float(x) if x != '' else None for x in line[1:]]

    def set_features(self, features: np.ndarray):
        """
        Sets the features of the molecule.

        :param features: A 1-D numpy array of features for the molecule.
        """
        self.features = features

    def num_tasks(self) -> int:
        """
        Returns the number of prediction tasks.

        :return: The number of tasks.
        """
        return len(self.targets)

    def set_targets(self, targets: List[float]):
        """
        Sets the targets of a molecule.

        :param targets: A list of floats containing the targets.
        """
        self.targets = targets


class MoleculeDataset(Dataset):
    """A MoleculeDataset contains a list of molecules and their associated features and targets."""

    def __init__(self, posdataa: List[MoleculeDatapoint], posdatab: List[MoleculeDatapoint], negdata: List[MoleculeDatapoint], negtypes: List[str], rels: List[int]):
        """
        Initializes a MoleculeDataset, which contains a list of MoleculeDatapoints (i.e. a list of molecules).

        :param data: A list of MoleculeDatapoints.
        """
        self.posdataa = posdataa
        self.posdatab = posdatab
        self.negdata = negdata
        self.rels = rels
        self.negtypes = [1 if negtype == 'h' else 0 for negtype in negtypes]
        self.args = self.posdataa[0].args if len(self.posdataa) > 0 else None
        self.scalera = None
        self.scalerb = None
        self.negscaler = None

    def compound_names(self) -> List[List[str]]:
        """
        Returns the compound names associated with the molecule (if they exist).

        :return: A list of compound names or None if the dataset does not contain compound names.
        """
        if len(self.posdataa) == 0 or self.posdataa[0].compound_name is None:
            return None

        return [d.compound_name for d in self.posdataa], [d.compound_name for d in self.posdatab], [d.compound_name for d in self.negdata]

    def smiles(self) -> List[List[str]]:
        """
        Returns the smiles strings associated with the molecules.

        :return: A list of smiles strings.
        """
        return [d.smiles for d in self.posdataa], [d.smiles for d in self.posdatab], [d.smiles for d in self.negdata]

    def features(self) -> List[List[np.ndarray]]:
        """
        Returns the features associated with each molecule (if they exist).

        :return: A list of 1D numpy arrays containing the features for each molecule or None if there are no features.
        """
        if len(self.posdataa) == 0 or self.posdataa[0].features is None:
            return None

        return [d.features for d in self.posdataa], [d.features for d in self.posdatab], [d.features for d in self.negdata]

    def targets(self) -> List[List[float]]:
        """
        Returns the targets associated with each molecule.

        :return: A list of lists of floats containing the targets.
        """
        return [d.targets for d in self.posdataa], [d.targets for d in self.posdatab], [d.targets for d in self.negdata]

    def features_size(self) -> int:
        """
        Returns the size of the features array associated with each molecule.

        :return: The size of the features.
        """
        return len(self.posdataa[0].features) if len(self.posdataa) > 0 and self.posdataa[0].features is not None else None

    def shuffle(self, seed: int = None):
        """
        Shuffles the dataset.

        :param seed: Optional random seed.
        """
        if seed is not None:
            random.seed(seed)
        random.shuffle(self.posdataa)
        random.shuffle(self.posdatab)
        random.shuffle(self.negdata)

    def normalize_features(self, scaler: StandardScaler = None, replace_nan_token: int = 0) -> List[StandardScaler]:
        """
        Normalizes the features of the dataset using a StandardScaler (subtract mean, divide by standard deviation).

        If a scaler is provided, uses that scaler to perform the normalization. Otherwise fits a scaler to the
        features in the dataset and then performs the normalization.

        :param scaler: A fitted StandardScaler. Used if provided. Otherwise a StandardScaler is fit on
        this dataset and is then used.
        :param replace_nan_token: What to replace nans with.
        :return: A fitted StandardScaler. If a scaler is provided, this is the same scaler. Otherwise, this is
        a scaler fit on this dataset.
        """
        if len(self.posdataa) == 0 or self.posdataa[0].features is None:
            return None
        for Scaler, data in zip((self.scalera, self.scalerb, self.negscaler),(self.posdataa, self.posdatab, self.negdata)):
            if scaler is not None:
                Scaler = scaler
            elif Scaler is None:
                features = np.vstack([d.features for d in data])
                Scaler = StandardScaler(replace_nan_token=replace_nan_token)
                Scaler.fit(features)

            for d in self.data:
                d.set_features(Scaler.transform(d.features.reshape(1, -1))[0])

        return self.scalera, self.scalerb, self.negscaler

    def set_targets(self, targets: List[List[float]]):
        """
        Sets the targets for each molecule in the dataset. Assumes the targets are aligned with the datapoints.

        :param targets: A list of lists of floats containing targets for each molecule. This must be the
        same length as the underlying dataset.
        """
        assert len(self.posdataa) == len(targets)
        for data in (self.posdataa, self.posdatab, self.negdata):
            for i in range(len(data)):
                data[i].set_targets(targets[i])

    def sort(self, key: Callable):
        """
        Sorts the dataset using the provided key.

        :param key: A function on a MoleculeDatapoint to determine the sorting order.
        """
        for data in (self.posdataa, self.posdatab, self.negdata):
            data.sort(key=key)

    def __len__(self) -> int:
        """
        Returns the length of the dataset (i.e. the number of molecules).

        :return: The length of the dataset.
        """
        return len(self.posdataa)

    def __getitem__(self, idx) -> List[Union[MoleculeDatapoint, List[MoleculeDatapoint]]]:
        """
        Gets one or more MoleculeDatapoints via an index or slice.

        :param item: An index (int) or a slice object.
        :return: A MoleculeDatapoint if an int is provided or a list of MoleculeDatapoints if a slice is provided.
        """
        return self.posdataa[idx], self.posdatab[idx], self.negdata[idx], self.negtypes[idx], self.rels[idx]

def filter_invalid_smiles(data: List[MoleculeDatapoint]) -> List[MoleculeDatapoint]:
    """
    Filters out invalid SMILES.

    :param data: A MoleculeDataset.
    :return: A MoleculeDataset with only valid molecules.
    """
    datapoint_list = []
    for idx, datapoint in enumerate(data):
        if datapoint.smiles == '':
            print(f'invalid smiles {idx}: {datapoint.smiles}')
            continue
        mol = Chem.MolFromSmiles(datapoint.smiles)
        if mol.GetNumHeavyAtoms() == 0:
            print(f'invalid heavy {idx}')
            continue
        datapoint_list.append(datapoint)
    return datapoint_list

def load_features(path: str) -> np.ndarray:
    """
    Loads features saved in a variety of formats.

    Supported formats:
    - .npz compressed (assumes features are saved with name "features")

    All formats assume that the SMILES strings loaded elsewhere in the code are in the same
    order as the features loaded here.

    :param path: Path to a file containing features.
    :return: A 2D numpy array of size (num_molecules, features_size) containing the features.
    """
    extension = os.path.splitext(path)[1]
    if extension == '.npz':
        features = np.load(path)['features']
    else:
        raise ValueError(f'Features path extension {extension} not supported.')

    return features

def get_data(path: str,
             smiles_path: str,
             skip_invalid_smiles: bool = True,
             args: Namespace = None,
             features_path: List[str] = None,
             max_data_size: int = None,
             use_compound_names: bool = None,
             logger: Logger = None) -> MoleculeDataset:
    """
    Gets smiles string and target values (and optionally compound names if provided) from a CSV file.

    :param path: Path to a CSV file.
    :param skip_invalid_smiles: Whether to skip and filter out invalid smiles.
    :param args: Arguments.
    :param features_path: A list of paths to files containing features. If provided, it is used
    in place of args.features_path.
    :param max_data_size: The maximum number of data points to load.
    :param use_compound_names: Whether file has compound names in addition to smiles strings.
    :param logger: Logger.
    :return: A MoleculeDataset containing smiles strings and target values along
    with other info such as additional features and compound names when desired.
    """
    debug = logger.debug if logger is not None else print

    if args is not None:
        # Prefer explicit function arguments but default to args if not provided
        features_path = features_path if features_path is not None else args.features_path
        max_data_size = max_data_size if max_data_size is not None else args.max_data_size
        use_compound_names = use_compound_names if use_compound_names is not None else args.use_compound_names
    else:
        use_compound_names = False

    max_data_size = max_data_size or float('inf')

    # Load features
    if features_path is not None:
        # for feat_path in features_path:
        features_data = load_features(features_path) # each is num_data x num_features
        args.features_dim = len(features_data[0])
        # print(args.features_dim//3)
        pair_num = len(features_data) // 3
        posa_features_data, posb_features_data, neg_features_data = features_data[:pair_num], features_data[pair_num:2*pair_num], features_data[2*pair_num:]
    else:
        features_data = None
        if args is not None:
            args.features_dim = 0
    
    skip_smiles = set()

    all_smiles = {}
    for _, items in pd.read_csv(smiles_path).iterrows():
        all_smiles[items['drug_id']] = items['smiles']

    # Load data
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)  # skip header

        linesa = []
        linesb = []
        linesneg = []
        rels = []
        negtypes = []

        for line in reader:
            smilesa = all_smiles[line[0]]
            smilesb = all_smiles[line[1]]
            rels.append(int(line[2]))
            smilesneg, negtype = line[3].split('$')
            smilesneg = all_smiles[smilesneg]

            if smilesa in skip_smiles or smilesb in skip_smiles or smilesneg in skip_smiles:
                continue

            linesa.append([smilesa])
            linesb.append([smilesb])
            linesneg.append([smilesneg])
            negtypes.append(negtype)

            if max(len(linesa), len(linesb), len(linesneg)) >= max_data_size:
                break
        
        # for i, line in tqdm(enumerate(linesa), total=len(linesa), disable=True):
        #     print(posa_features_data[i])
        posdataa = [
            MoleculeDatapoint(
                line=line,
                args=args,
                features=posa_features_data[i] if posa_features_data is not None else None,
                use_compound_names=use_compound_names
            ) for i, line in tqdm(enumerate(linesa), total=len(linesa), disable=True)
        ]

        posdatab = [
            MoleculeDatapoint(
                line=line,
                args=args,
                features=posb_features_data[i] if posb_features_data is not None else None,
                use_compound_names=use_compound_names
            ) for i, line in tqdm(enumerate(linesb), total=len(linesb), disable=True)
        ]

        negdata = [
            MoleculeDatapoint(
                line=line,
                args=args,
                features=neg_features_data[i] if neg_features_data is not None else None,
                use_compound_names=use_compound_names
            ) for i, line in tqdm(enumerate(linesneg), total=len(linesneg), disable=True)
        ]

    # Filter out invalid SMILES
    if skip_invalid_smiles:
        for data in (posdataa, posdatab, negdata):
            original_data_len = len(data)
            data = filter_invalid_smiles(data)

            if len(data) < original_data_len:
                debug(f'Warning: {original_data_len - len(data)} SMILES are invalid.')

    return MoleculeDataset(posdataa, posdatab, negdata, negtypes, rels)