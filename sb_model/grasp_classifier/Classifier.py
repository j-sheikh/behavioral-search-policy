import torch
from torch import nn, optim
import torchmetrics
from pytorch_lightning.core.module import LightningModule
from CNNModel import CNNModel

class GraspClassifier(LightningModule):
    def __init__(self, num_classes, class_weights, batch_size=32, learning_rate=0.001):
        super().__init__()

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)

        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        self.cnn_model = CNNModel()

    def forward(self, x):
        return self.cnn_model(x)

    def _common_step(self, batch):
        x, y = batch
        scores = self.forward(x)
        loss = self.loss_fn(scores, y)
        return loss, scores, y


    def training_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch)
        accuracy = self.accuracy(scores, y)
        f1_score = self.f1_score(scores, y)
        self.log_dict(
            {
                "train_loss": loss,
                "train_accuracy": accuracy,
                "train_f1_score": f1_score,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return {"loss": loss, "scores": scores, "y": y}


    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch)
        self.log("test_loss", loss)
        return loss

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            output = self(x)
        _, predicted_label = torch.max(output, 1)
        predicted_label = predicted_label.cpu().item()
        predicted_probabilities = torch.softmax(output, dim=1).squeeze().tolist()
        return predicted_label, predicted_probabilities


    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, num_classes, class_weights):
        checkpoint = torch.load(checkpoint_path)
        model = cls(num_classes=num_classes, class_weights=class_weights)
        model.load_state_dict(checkpoint['state_dict'])

        return model

