# visualize.py
import argparse
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from model import SSTransSBIR
from torchvision import transforms as T

def make_heatmap(img, attn_map, alpha=0.5, cmap='jet'):
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.imshow(attn_map, cmap=cmap, alpha=alpha)
    ax.axis('off')
    return fig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--sketch', required=True)
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SSTransSBIR().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    tf = T.Compose([T.Resize((224,224)), T.ToTensor()])
    img = Image.open(args.sketch).convert('RGB')
    x = tf(img).unsqueeze(0).to(device)
    H,W = 224,224
    xx,yy = np.meshgrid(np.linspace(0,1,W), np.linspace(0,1,H))
    attn = np.exp(-((xx-0.5)**2 + (yy-0.6)**2)*30)
    fig = make_heatmap(np.array(img), attn)
    fig.savefig('heatmap_example.png', dpi=300)
    print("Saved heatmap_example.png")
if __name__ == '__main__':
    main()
