# eval.py
import argparse
import torch
from model import SSTransSBIR
from dataset import ChairV2Dataset
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
from tqdm import tqdm
import yaml

def extract_embeddings(model, loader, device='cpu'):
    model.eval()
    embs = []
    labels = []
    with torch.no_grad():
        for imgs, labs in tqdm(loader):
            img = imgs.to(device)
            emb = model(img, img)
            embs.append(emb.cpu().numpy())
            labels.extend(labs)
    embs = np.vstack(embs)
    return embs, np.array(labels)

def compute_cmc_precision(query_embs, gallery_embs, qlabels, glabels, topk=[1,5,10]):
    sims = cosine_similarity(query_embs, gallery_embs)
    ranks = np.argsort(-sims, axis=1)
    cmc = {}
    for k in topk:
        correct = 0
        for i in range(ranks.shape[0]):
            topk_inds = ranks[i,:k]
            if glabels[topk_inds].tolist().count(qlabels[i])>0:
                correct += 1
        cmc[k] = correct / ranks.shape[0]
    return cmc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_root', type=str, required=True)
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SSTransSBIR().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model'])
    ds = ChairV2Dataset(args.data_root)
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    embs, labels = extract_embeddings(model, loader, device=device)
    N = len(embs)//2
    q_embs, g_embs = embs[:N], embs[N:]
    q_labels, g_labels = labels[:N], labels[N:]
    cmc = compute_cmc_precision(q_embs, g_embs, q_labels, g_labels, topk=[1,5,10])
    print("CMC:", cmc)

if __name__ == '__main__':
    main()
