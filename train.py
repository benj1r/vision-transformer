import torch.nn.functional as F


class Trainer:
    def __init__(self, model, optimizer, dataset):
        self.model = model
        self.optimizer = optimizer
        self.dataset = dataset

    def train(self):        
        losses = []
        for batch_idx, (data, targets) in self.dataset:
            scores = self.model(data)
            
            self.optimizer.zero_grad()
            loss = F.nll_loss(scores, targets)
            losses.append(loss.item())
            loss.backward()
            self.optimizer.step()
            self.dataset.set_description(f"train | loss: { round(loss.item(), 2) }")
        
        # return training loss
        return losses
