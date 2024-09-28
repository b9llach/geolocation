import os
import csv
import random
import torch
import timm
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import OrderedDict
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import gc
import logging
from typing import List, Tuple, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LRUCache:
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key: str) -> Optional[Image.Image]:
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: str, value: Image.Image) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

class GeolocationDataset(Dataset):
    def __init__(self, csv_file: str, img_dir: str, transform: Optional[A.Compose] = None, max_samples: Optional[int] = None, cache_size: int = 10000):
        self.img_dir = img_dir
        self.transform = transform
        self.data: List[Tuple[str, float, float]] = []
        self.cache = LRUCache(cache_size)
        
        with open(csv_file, 'r') as f:
            csv_reader = csv.reader(f)
            next(csv_reader, None)  # Skip the header row
            for row in csv_reader:
                if len(row) == 3:
                    pano_id, lat, lng = row
                    try:
                        self.data.append((pano_id, float(lat), float(lng)))
                    except ValueError:
                        logging.warning(f"Skipping invalid row: {row}")
        
        random.shuffle(self.data)
        if max_samples and max_samples < len(self.data):
            self.data = self.data[:max_samples]
        
        logging.info(f"Loaded {len(self.data)} panorama entries")

    @staticmethod
    def normalize_coords(lat: float, lng: float) -> Tuple[float, float]:
        norm_lat = (lat + 90) / 180
        norm_lng = (lng + 180) / 360
        return norm_lat, norm_lng

    @staticmethod
    def denormalize_coords(norm_lat: float, norm_lng: float) -> Tuple[float, float]:
        lat = (norm_lat * 180) - 90
        lng = (norm_lng * 360) - 180
        return lat, lng

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        pano_id, lat, lng = self.data[idx]
        
        cached_image = self.cache.get(pano_id)
        if cached_image is not None:
            norm_lat, norm_lng = self.normalize_coords(lat, lng)
            return self.transform(image=cached_image)["image"], torch.tensor([norm_lat, norm_lng])
        
        for i in range(13):
            img_path = os.path.join(self.img_dir, f"{pano_id}_{i}.jpg")
            if os.path.exists(img_path):
                try:
                    image = Image.open(img_path).convert('RGB')
                    self.cache.put(pano_id, image)
                    if self.transform:
                        image = self.transform(image=image)["image"]
                    norm_lat, norm_lng = self.normalize_coords(lat, lng)
                    return image, torch.tensor([norm_lat, norm_lng])
                except (IOError, OSError):
                    logging.error(f"Error loading image: {img_path}")
        
        return None

class GeolocationModel(nn.Module):
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model
        self.regressor = nn.Sequential(
            nn.Linear(self.base_model.num_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.base_model(x)
        return self.regressor(features)

def haversine_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    R = 6371  # Earth's radius in kilometers

    lat1, lon1 = y_pred[:, 0], y_pred[:, 1]
    lat2, lon2 = y_true[:, 0], y_true[:, 1]

    lat1, lon1, lat2, lon2 = map(torch.deg2rad, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = torch.sin(dlat/2)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon/2)**2
    c = 2 * torch.asin(torch.sqrt(a))

    distance = R * c

    return torch.mean(distance)

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

def calculate_distance_error(outputs, labels, dataset):
    outputs = torch.tensor([dataset.denormalize_coords(lat, lng) for lat, lng in outputs])
    labels = torch.tensor([dataset.denormalize_coords(lat, lng) for lat, lng in labels])
    return haversine_loss(outputs, labels).item()

def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, criterion: callable, optimizer: torch.optim.Optimizer, num_epochs: int = 50, dataset: Dataset = None, device: torch.device = None, accumulation_steps: int = 8):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = torch.amp.GradScaler('cuda')

    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        gc.collect()
        
        model.train()
        train_loss = 0
        optimizer.zero_grad()
        for i, (inputs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training")):
            inputs, labels = inputs.to(device), labels.to(device)
            
            with autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss = loss / accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            train_loss += loss.item() * accumulation_steps

        model.eval()
        val_loss = 0
        all_outputs = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
                all_outputs.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        distance_error = calculate_distance_error(torch.tensor(all_outputs), torch.tensor(all_labels), dataset.dataset)
        
        logging.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        logging.info(f"Mean Distance Error: {distance_error:.2f} km")
        
        scheduler.step()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            if isinstance(model, nn.parallel.DistributedDataParallel):
                torch.save(model.module.state_dict(), f'best_geolocation_model.pth')
            else:
                torch.save(model.state_dict(), f'best_geolocation_model.pth')
            logging.info(f"New best model saved as 'best_geolocation_model.pth'")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f"Early stopping triggered after {epoch+1} epochs")
                break

    return model

def setup(rank: int, world_size: int):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train_model_ddp(rank: int, world_size: int, model: nn.Module, train_dataset: Dataset, val_dataset: Dataset, num_epochs: int = 50):
    setup(rank, world_size)
    
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    
    model = model.to(device)
    model = DDP(model, device_ids=[rank])
    
    batch_size = 32
    num_workers = 4
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)
    
    criterion = haversine_loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    
    try:
        train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=num_epochs, dataset=train_dataset, device=device, accumulation_steps=4)
    except Exception as e:
        logging.error(f"Rank {rank} encountered an error: {str(e)}")
    finally:
        cleanup()

def create_ensemble(num_models: int = 3) -> List[nn.Module]:
    models = []
    for _ in range(num_models):
        base_model = timm.create_model('vit_large_patch16_224', pretrained=True, num_classes=0)
        model = GeolocationModel(base_model)
        models.append(model)
    return models

def ensemble_predict(models: List[nn.Module], inputs: torch.Tensor) -> torch.Tensor:
    predictions = []
    for model in models:
        model.eval()
        with torch.no_grad():
            pred = model(inputs)
        predictions.append(pred)
    return torch.stack(predictions).mean(dim=0)

def main():
    csv_file = '/data/geolocation/panos/panorama_log.csv'
    img_dir = '/data/geolocation/panos_only_mid'
    
    transform = A.Compose([
        A.RandomResizedCrop(224, 224, scale=(0.8, 1.0)),
        A.HorizontalFlip(),
        A.RandomRotate90(),
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    full_dataset = GeolocationDataset(csv_file, img_dir, transform=transform, cache_size=50000)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    ensemble_models = create_ensemble(num_models=3)

    world_size = torch.cuda.device_count()
    for i, model in enumerate(ensemble_models):
        mp.spawn(
            train_model_ddp,
            args=(world_size, model, train_dataset, val_dataset, 50),
            nprocs=world_size,
            join=True
        )
        
        torch.save(model.state_dict(), f'geolocation_model_ensemble_{i+1}.pth')

    logging.info("Training complete.")

if __name__ == "__main__":
    main()