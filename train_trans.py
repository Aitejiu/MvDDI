from datetime import datetime
import pdb
import time
from models import MvModel
from grover.util.parsing import *
from dataset import *
from molgraph import *
from custom_loss import SigmoidLoss
from torch import optim
from custom_metrics import do_compute_metrics
import logging

torch.multiprocessing.set_sharing_strategy('file_system')
args = parse_args()

if args.dataset != 'twosides':
    args.rels = 86
else:
    args.rels = 963

if args.data_dir == None:
    args.data_dir = 'data'

args.smiles_model_path = 'model/MoLFormer-XL-both-10pct'
args.mol_model_path = 'model/grover_large.pt'
args.dropout = 0.1
args.train_data_path = f'{args.data_dir}/{args.dataset}/fold{args.fold}/train.csv'
args.train_features_path = f'{args.data_dir}/{args.dataset}/fold{args.fold}/train_smiles.npz'
args.test_data_path = f'{args.data_dir}/{args.dataset}/fold{args.fold}/test.csv'
args.test_features_path = f'{args.data_dir}/{args.dataset}/fold{args.fold}/test_smiles.npz'
args.valid_data_path = f'{args.data_dir}/{args.dataset}/fold{args.fold}/valid.csv'
args.valid_features_path = f'{args.data_dir}/{args.dataset}/fold{args.fold}/valid_smiles.npz'
args.smiles_path = f'{args.data_dir}/{args.dataset}/drug_smiles.csv'


random.seed(2024)
dataset_name = args.dataset
fold_i = args.fold
dropout = args.dropout
batch_size = args.batch_size
if args.device == None:
    args.device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
args.hid_feats = 64
args.lr = 3e-5
args.weight_decay = 5e-7
args.n_epochs = 100

log_path = f'log/{args.dataset}/{datetime.now().strftime("%m_%d_%H_%M_%S")}.log'
if os.path.exists(os.path.dirname(log_path)) == False:
    os.makedirs(os.path.dirname(log_path))
logger = logging.getLogger('train')
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(log_path)
file_handler.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
formater = logging.Formatter("%(asctime)s %(name)s:%(levelname)s:%(message)s", datefmt="%d-%M-%Y %H:%M:%S")
file_handler.setFormatter(formater)
stream_handler.setFormatter(formater)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)
args.logger = logger
args.model_save = f'./model/{args.dataset}/{datetime.now().strftime("%m_%d_%H_%M_%S")}/'
if os.path.exists(args.model_save) == False:
    os.makedirs(args.model_save)
result_save = f'./table/{args.dataset}/{datetime.now().strftime("%m_%d_%H_%M_%S")}.csv'
if os.path.exists(os.path.dirname(result_save)) == False:
    os.makedirs(os.path.dirname(result_save))
result = pd.DataFrame({'loss':[],'acc':[],'roc':[],'f1':[], 'p':[], 'r':[], 'int-ap':[], 'ap':[]})

logger.info(args)

def do_compute(model, batch, device): 
        p_score, n_score = model(batch, device)
        probas_pred = np.concatenate([torch.sigmoid(p_score.detach()).cpu().mean(dim=-1), torch.sigmoid(n_score.detach()).cpu().mean(dim=-1)])
        ground_truth = np.concatenate([np.ones(p_score.shape[0]), np.zeros(n_score.shape[0])])

        return p_score, n_score, probas_pred, ground_truth

def run_batch(model, optimizer, data_loader, epoch_i, desc, loss_fn, device):
        total_loss = 0
        loss_pos = 0
        loss_neg = 0
        probas_pred = []
        ground_truth = []
        for batch in tqdm(data_loader, desc= f'{desc} Epoch {epoch_i}', ascii = False):
            p_score, n_score, batch_probas_pred, batch_ground_truth = do_compute(model, batch, device)
            probas_pred.append(batch_probas_pred)
            ground_truth.append(batch_ground_truth)
            loss, loss_p, loss_n = loss_fn(p_score, n_score)
            if model.training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            loss_pos += loss_p.item()
            loss_neg += loss_n.item() 
        total_loss /= len(data_loader)
        loss_pos /= len(data_loader)
        loss_neg /= len(data_loader)
        
        probas_pred = np.concatenate(probas_pred)
        ground_truth = np.concatenate(ground_truth)     
        return total_loss, do_compute_metrics(probas_pred, ground_truth)

def print_metrics(loss, acc, auroc, f1_score, precision, recall, int_ap, ap):
    result.loc[len(result)] = [loss,acc,auroc,f1_score,precision,recall,int_ap, ap]
    result.to_csv(result_save)
    logger.info(f'loss: {loss:.4f}, acc: {acc:.4f}, roc: {auroc:.4f}, f1: {f1_score:.4f},')
    logger.info(f'p: {precision:.4f}, r: {recall:.4f}, int-ap: {int_ap:.4f}, ap: {ap:.4f}')  

    return f1_score

def test(model, test_data_loader, loss_fn, device):
    model.eval()
    with torch.no_grad():
        test_loss, test_metrics = run_batch(model, None, test_data_loader, 0, 'test', loss_fn, device)
        logger.info('#### Test')
        print_metrics(test_loss, *test_metrics)

def train(model, train_data_loader, val_data_loader, test_data_loader, loss_fn, optimizer, n_epochs, device, scheduler):
    bestacc = 0
    update_num = 0
    lt = None
    for epoch_i in range(1, n_epochs+1):
        with torch.cuda.device(args.device):
            torch.cuda.empty_cache()  # 释放显存
        start = time.time()
        model.train()
        ## Training
        train_loss, train_metrics = run_batch(model, optimizer, train_data_loader, epoch_i,  'train', loss_fn, device)
        if scheduler:
            scheduler.step()
        with torch.cuda.device(args.device):
            torch.cuda.empty_cache()  # 释放显存
        model.eval()
        with torch.no_grad():

            ## Validation 
            if val_data_loader:
                val_loss , val_metrics = run_batch(model, optimizer, val_data_loader, epoch_i, 'val', loss_fn, device)

        if train_data_loader:
            logger.info(f'#### Epoch{epoch_i} Epoch time {time.time() - start:.4f}s')
            logger.info('#### Train')
            print_metrics(train_loss, *train_metrics)
        
        if val_data_loader:
            logger.info('#### Validation')
            print_metrics(val_loss, *val_metrics)
        
        if val_metrics[0] > bestacc:
                bestacc = val_metrics[0]
                path = f'{args.model_save}{datetime.now().strftime("%m_%d_%H_%M_%S")}/lr_{args.lr}_bs_{batch_size}_ep_{epoch_i}_model.pth'
                if not os.path.exists(path):
                    os.makedirs(os.path.dirname(path))
                torch.save(model.state_dict(),path)
                logger.info(f'best model saved to {path}')
                lt = path
                update_num = 0

        update_num += 1
        if update_num > 10 and lt != None:
            break
    
    model.load_state_dict(torch.load(lt, map_location=torch.device(args.device)))
    test(model, test_data_loader, loss_fn, device)

train_data = get_data(path=args.train_data_path,
                     smiles_path=args.smiles_path,
                     features_path=args.train_features_path,
                     args=args,
                     use_compound_names=False,
                     max_data_size=float("inf"),
                     skip_invalid_smiles=False)

valid_data = get_data(path=args.valid_data_path,
                     smiles_path=args.smiles_path,
                     features_path=args.valid_features_path,
                     args=args,
                     use_compound_names=False,
                     max_data_size=float("inf"),
                     skip_invalid_smiles=False)

test_data = get_data(path=args.test_data_path,
                     smiles_path=args.smiles_path,
                     features_path=args.test_features_path,
                     args=args,
                     use_compound_names=False,
                     max_data_size=float("inf"),
                     skip_invalid_smiles=False)

logger.info(f'Total train size = {len(train_data):,}')
logger.info(f'Total valid size = {len(valid_data):,}')
logger.info(f'Total test size = {len(test_data):,}')

mol_collator = MolCollator(args=args, shared_dict={})

num_workers = 4
train_mol_loader = DataLoader(train_data,
                        batch_size=args.batch_size,
                        shuffle=True,
                        num_workers=num_workers,
                        collate_fn=mol_collator)

val_mol_loader = DataLoader(valid_data,
                        batch_size=args.batch_size,
                        shuffle=True,
                        num_workers=num_workers,
                        collate_fn=mol_collator)

test_mol_loader = DataLoader(test_data,
                        batch_size=args.batch_size,
                        shuffle=True,
                        num_workers=num_workers,
                        collate_fn=mol_collator)

model = MvModel(args).to(args.device)
if args.model_path:
    model.load_state_dict(torch.load(args.model_path, map_location=torch.device(args.device)))
loss_fn = SigmoidLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.96 ** (epoch))

time_stamp = f'{datetime.now()}'.replace(':', '_')

if args.test:
    logger.info(f'Testing on {args.device}.')
    logger.info(f'Starting fold_{fold_i} at')
    test(model, test_mol_loader, loss_fn, args.device)
else:
    logger.info(f'Training on {args.device}.')
    logger.info(f'Starting fold_{fold_i} at')
    train(model, train_mol_loader, val_mol_loader, test_mol_loader, loss_fn, optimizer, args.n_epochs, args.device, scheduler)