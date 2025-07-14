import pdb
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from grover.util.utils import load_checkpoint

class SmilesEncoder(nn.Module):
    def __init__(self, Textencoder_path, args):
        super().__init__()
        self.Smilesencoder = AutoModel.from_pretrained(Textencoder_path, deterministic_eval=True, trust_remote_code=True).to(args.device)
        self.SmilesTokenizer = AutoTokenizer.from_pretrained(Textencoder_path, trust_remote_code=True)
    
    def forward(self, pos_smilesa, pos_smilesb, neg_smiles, device):        
        # get pos drug featrues
        pos_smilesa_features = self.Smilesencoder(**self.SmilesTokenizer(pos_smilesa, padding=True, return_tensors="pt").to(device)).last_hidden_state
        pos_smilesb_features = self.Smilesencoder(**self.SmilesTokenizer(pos_smilesb, padding=True, return_tensors="pt").to(device)).last_hidden_state

        # get neg drug features
        neg_smiles_features = self.Smilesencoder(**self.SmilesTokenizer(neg_smiles, padding=True, return_tensors="pt").to(device)).last_hidden_state

        return pos_smilesa_features, pos_smilesb_features, neg_smiles_features

class CustomDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.dropout = (lambda x: x ) if p == 0 else nn.Dropout(p)
    
    def forward(self, input):
        return self.dropout(input)

class Decoder(nn.Module):
    def __init__(self, d_features, hid_feats, dropout):
        super().__init__()
        self.d_features = d_features
        self.hid_feats = hid_feats
        self.dropout = dropout
        self.decoder = nn.Sequential(
            nn.Linear(self.d_features, 4 * self.hid_feats),
            nn.LayerNorm(4 * self.hid_feats),
            nn.ELU(),
            CustomDropout(self.dropout),
            nn.Linear(4 * self.hid_feats, 2 * self.hid_feats),
            nn.LayerNorm(2 * self.hid_feats),
            nn.ELU(),
            CustomDropout(self.dropout),
            nn.Linear(2 * self.hid_feats, self.hid_feats),
            nn.LayerNorm(self.hid_feats)
        )
    
    def forward(self, batch):
        return self.decoder(batch)

class FFN(nn.Module):
    def __init__(self, d_model, d_hid, dropout=0, d_out=1):
        super().__init__()
        self.W_1 = nn.Linear(d_model, 4 * d_hid)
        self.W_2 = nn.Linear(4 * d_hid, 2 * d_hid)
        self.W_3 = nn.Linear(2 * d_hid, d_out)
        # self.norm = nn.BatchNorm1d(d_hid)
        self.dropout = CustomDropout(dropout)
        self.act_func = nn.ELU()

    def forward(self, x):
        res1 = self.W_2(self.dropout(self.act_func(self.W_1(x))))
        return self.W_3(self.dropout(self.act_func(res1)))

class CoAttentionLayer(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.n_features = n_features
        self.w_q = nn.Parameter(torch.zeros(n_features, n_features//2))
        self.w_k = nn.Parameter(torch.zeros(n_features, n_features//2))
        self.bias = nn.Parameter(torch.zeros(n_features // 2))
        self.a = nn.Parameter(torch.zeros(n_features//2))

        nn.init.xavier_uniform_(self.w_q)
        nn.init.xavier_uniform_(self.w_k)
        nn.init.xavier_uniform_(self.bias.view(*self.bias.shape, -1))
        nn.init.xavier_uniform_(self.a.view(*self.a.shape, -1))
    
    def forward(self, receiver, attendant):
        keys = receiver @ self.w_k
        queries = attendant @ self.w_q
        values = receiver

        e_activations = queries.unsqueeze(-3) + keys.unsqueeze(-2) + self.bias
        e_scores = torch.tanh(e_activations) @ self.a
        attentions = e_scores
        return attentions

class RESCAL(nn.Module):
    def __init__(self, n_rels, n_features):
        super().__init__()
        self.n_rels = n_rels
        self.n_features = n_features
        self.rel_emb = nn.Embedding(self.n_rels, n_features * n_features)
        self.a2b_mlp = FFN(256, n_features, d_out=n_features)
        self.b2a_mlp = FFN(256, n_features, d_out=n_features)
        nn.init.xavier_uniform_(self.rel_emb.weight)
    
    def forward(self, heads, tails, rels, alpha_scores):
        rels = self.rel_emb(rels)
        rels = F.normalize(rels, dim=-1)

        heads = F.normalize(heads, dim=-1)
        tails = F.normalize(tails, dim=-1)
        rels = rels.view(-1, self.n_features, self.n_features)

        scores = heads @ rels @ tails.transpose(-2, -1)

        if alpha_scores is not None:
          scores = alpha_scores * scores
        a2b = scores.sum(dim=-1)
        b2a = scores.sum(dim=-2)
        scores = scores.sum(dim=(-2, -1))
        if a2b.shape[1] >= 256:
            a2b = a2b[:, :256]
        else:
            a2b = torch.concat([a2b, scores.unsqueeze(-1).repeat(1, 256 - a2b.shape[1])], dim=-1)
        if b2a.shape[1] >= 256:
            b2a = b2a[:, :256]
        else:
            b2a = torch.cat([b2a, scores.unsqueeze(-1).repeat(1, 256 - b2a.shape[1])], dim=-1)
        scores = torch.cat([self.a2b_mlp(a2b), self.b2a_mlp(b2a)], dim=-1)
        return scores 
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.n_rels}, {self.rel_emb.weight.shape})"

class MvModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.smiles_encoder = SmilesEncoder(args.smiles_model_path, args)
        self.Mol_encoder = load_checkpoint(args.mol_model_path, cuda=args.cuda, current_args=args)
        self.hid_features = args.hid_features
        self.batch_size = args.batch_size
        self.dropout = args.dropout
        self.d_smiles = 768
        self.d_mol = 2400
        self.features_dim = args.features_dim
        self.smiles_decoder = Decoder(self.d_smiles, self.hid_features, self.dropout)
        self.mol_atom_decoder = Decoder(self.d_mol, self.hid_features, self.dropout)
        self.mol_bond_decoder = Decoder(self.d_mol, self.hid_features, self.dropout)
        self.coattention = CoAttentionLayer(self.hid_features)
        self.KGE = RESCAL(args.rels, self.hid_features)
        self.FFN = FFN(self.features_dim * 2 + self.hid_features * 8, self.hid_features, self.dropout)
        
    def compute_score(self, pos_a_features, pos_b_features, neg_a_features, neg_b_features, rels):
        pos_attentions = self.coattention(pos_a_features, pos_b_features)
        neg_attentions = self.coattention(neg_a_features, neg_b_features)
        p_score = self.KGE(pos_a_features, pos_b_features, rels, pos_attentions)
        n_score = self.KGE(neg_a_features, neg_b_features, rels, neg_attentions)
        return p_score, n_score

    def forward(self, batch, device):
        possmilesa_batch, possmilesb_batch, negsmiles_batch, batcha, batchb, negbatch, posfeaturesa_batch, posfeaturesb_batch, negfeatures_batch, negtypes, rels = batch
        # smiles_features
        pos_smilesa_features, pos_smilesb_features, neg_smiles_features = self.smiles_encoder(possmilesa_batch, possmilesb_batch, negsmiles_batch, device)
        pos_smilesa_features = self.smiles_decoder(pos_smilesa_features)
        pos_smilesb_features = self.smiles_decoder(pos_smilesb_features)
        neg_smiles_features = self.smiles_decoder(neg_smiles_features)
        # mol_features
        pos_mola_atom_features, pos_mola_bond_features, pos_a_mor_features= self.Mol_encoder(batcha, posfeaturesa_batch)
        pos_molb_atom_features, pos_molb_bond_features, pos_b_mor_features= self.Mol_encoder(batchb, posfeaturesb_batch)
        neg_mol_atom_features, neg_mol_bond_features, neg_mor_features= self.Mol_encoder(negbatch, negfeatures_batch)
        pos_mola_atom_features = self.mol_atom_decoder(pos_mola_atom_features)
        pos_molb_atom_features = self.mol_atom_decoder(pos_molb_atom_features)
        neg_mol_atom_features = self.mol_atom_decoder(neg_mol_atom_features)
        pos_mola_bond_features = self.mol_bond_decoder(pos_mola_bond_features)
        pos_molb_bond_features = self.mol_bond_decoder(pos_molb_bond_features)
        neg_mol_bond_features = self.mol_bond_decoder(neg_mol_bond_features)
        pos_mola_features = torch.cat([pos_mola_atom_features, pos_mola_bond_features], dim=-2)
        pos_molb_features = torch.cat([pos_molb_atom_features, pos_molb_bond_features], dim=-2)
        neg_mol_features = torch.cat([neg_mol_atom_features, neg_mol_bond_features], dim=-2)
        
        if negtypes == 1:
            neg_mola_features = neg_mol_features.clone()
            neg_smilesa_features = neg_smiles_features.clone()
            neg_molb_features = pos_molb_features.clone()
            neg_smilesb_features = pos_smilesb_features.clone()
        else:
            neg_mola_features = pos_mola_features.clone()
            neg_smilesa_features = pos_smilesa_features.clone()
            neg_molb_features = neg_mol_features.clone()
            neg_smilesb_features = neg_smiles_features.clone()
        rels = rels.to(device)

        # smi-smi
        smi_smi_p_score, smi_smi_n_score = self.compute_score(pos_smilesa_features, pos_smilesb_features, neg_smilesa_features, neg_smilesb_features, rels)
        # mol-mol
        mol_mol_p_score, mol_mol_n_score = self.compute_score(pos_mola_features, pos_molb_features, neg_mola_features, neg_molb_features, rels)
        # mol-smi
        mol_smi_p_score, mol_smi_n_score = self.compute_score(pos_mola_features, pos_smilesb_features, neg_mola_features, neg_smilesb_features, rels)
        # smi-mol
        smi_mol_p_score, smi_mol_n_score = self.compute_score(pos_smilesa_features, pos_molb_features, neg_smilesa_features, neg_molb_features, rels)
        p_score = self.FFN(torch.cat([pos_a_mor_features, pos_b_mor_features, smi_smi_p_score, smi_mol_p_score, mol_mol_p_score, mol_smi_p_score], dim=-1))
        if negtypes == 1:
            n_score = self.FFN(torch.cat([neg_mor_features, pos_b_mor_features, smi_smi_n_score, smi_mol_n_score, mol_mol_n_score, mol_smi_n_score], dim=-1))
        else:
            n_score = self.FFN(torch.cat([neg_mor_features, pos_a_mor_features, smi_smi_n_score, smi_mol_n_score, mol_mol_n_score, mol_smi_n_score], dim=-1))
        return p_score, n_score