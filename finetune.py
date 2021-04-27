from data import DISFADataset, collate_missing, cycle, DISFA_get_videos_tvt
from transformer import FaceTransformer, CosineWithRestarts
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import time
import sys

timestamp = str(int(time.time()))

root_dir = "/media/DISFA/FaceKeypoints/"
labels_dir = "/media/DISFA/Labels"
save_dir = "/media/face-transformer/checkpoints/"

train, val, test = DISFA_get_videos_tvt(root_dir)
print(train, val, test)

sys.exit(0)

dataset = PretrainingDataset(root_dir)
dataloader = cycle(DataLoader(
    dataset,
    batch_size=350,
    shuffle=True,
    num_workers=16,
    collate_fn=collate_missing
))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FaceTransformer(2278).to(device)
#model = torch.nn.DataParallel(model).to(device)

criterion = torch.nn.MSELoss()
num_batches = 100000
accum_steps = 1
save_every = 1000
restart_every = 5000
lr = 1e-6
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
sched = CosineWithRestarts(optimizer, T_max=restart_every+1)

for i in range(num_batches):
    model.train()
    batch_loss = 0
    for __ in range(accum_steps):
        arrays, seq_idxs, seq_label_idxs, label_arrays = next(dataloader)
        preds = model(arrays.to(device), seq_idxs.to(device), seq_label_idxs.to(device))
        loss = criterion(preds.cpu(), label_arrays)
        batch_loss += loss.item()
        loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()
    sched.step()
    optimizer.zero_grad()

    if i == 0 or (((i+1) % save_every) == 0):
        torch.save(model.state_dict(), Path(save_dir, F"fnt-DISFA-{timestamp}-batch-{i+1}.ckpt"))

    print(F"batch {i}, loss {batch_loss}")
