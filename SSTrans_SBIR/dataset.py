# dataset.py
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

class ChairV2Dataset(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.transform = transform or T.Compose([
            T.Resize((224,224)),
            T.ToTensor()
        ])
        self.items = []
        for cls in sorted(os.listdir(root)):
            cls_dir = os.path.join(root, cls)
            if not os.path.isdir(cls_dir): continue
            for fn in os.listdir(cls_dir):
                if fn.lower().endswith(('.jpg','.png')):
                    self.items.append((os.path.join(cls_dir, fn), cls))
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        path, cls = self.items[idx]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        label = int(cls) if cls.isdigit() else cls
        return img, label

def make_dataloader(root, batch_size=32, shuffle=True, num_workers=4):
    ds = ChairV2Dataset(root)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
