# train.py
import argparse
import torch
from torch import optim
from torch.utils.data import DataLoader
from model import SSTransSBIR
from dataset import make_dataloader
from losses import TripletLoss, AlignmentLoss
from tqdm import tqdm
import os
import yaml

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--data_root', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--save_dir', type=str, default='checkpoints/')
    return parser.parse_args()

def load_config(path):
    with open(path,'r') as f:
        return yaml.safe_load(f)

def main():
    args = parse_args()
    cfg = load_config(args.config)
    data_root = args.data_root or cfg['dataset']['path']
    epochs = args.epochs or cfg['training']['epochs']
    batch_size = args.batch_size or cfg['training']['batch_size']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = SSTransSBIR(d_model=cfg['model']['d_model']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr or cfg['training']['lr'])
    triplet = TripletLoss(margin=0.2)
    align = AlignmentLoss()

    train_loader = make_dataloader(data_root, batch_size=batch_size, shuffle=True)
    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(1, epochs+1):
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for imgs, labels in pbar:
            imgs = imgs.to(device)
            if imgs.size(0) < 2: continue
            half = imgs.size(0)//2
            sketch = imgs[:half]
            photo = imgs[half:half*2]
            emb_s = model(sketch, photo)
            emb_p = model(photo, sketch)
            pos = emb_p
            neg = torch.roll(emb_p, shifts=1, dims=0)
            l_trip = triplet(emb_s, pos, neg)
            l_align = align(emb_s, pos)
            loss = l_trip + 0.3 * l_align
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{epoch_loss/ (pbar.n+1):.4f}'})
        torch.save({'model': model.state_dict(), 'opt': optimizer.state_dict()}, os.path.join(args.save_dir, f'epoch_{epoch}.pth'))
    print("Training finished.")

if __name__ == '__main__':
    main()
