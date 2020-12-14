import torch
from src import datasets, models
import os

def trainval():

    print("train")
    num_epochs = 1
    results = {}
    train_dl = datasets.get_dataset(dataroot="test_ds", image_size=64, batch_size=128, num_workers=2)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = models.Model(device)
    
    score_list = []
    for i in range(0, num_epochs):
        lossD, lossG = model.train_on_dataset(train_dl)
        results["lossD"] = lossD
        results["lossG"] = lossG
        model.vis_on_dataset(fname=os.path.join('training_image_results', 'results.png'))

if __name__ == "__main__":
    trainval()
