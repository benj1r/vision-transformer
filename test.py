import torch.nn.functional as F


class Tester:
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset

    def test(self):
        losses = []
        for batch_idx, (data, targets) in self.dataset:
            scores = self.model(data)
            
            loss = F.nll_loss(scores, targets)
            losses.append(loss.item())
            loss.backward()
            
            self.dataset.set_description(f"test  | loss: { round(loss.item(), 2) }")
            

        # return testing loss
        return losses
