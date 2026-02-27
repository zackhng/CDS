import os
import torch
import torch.nn as nn
from torch import Tensor

class VGG16Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_of_conv: int):
        super().__init__()
        layers = []
        for i in range(num_of_conv):
            layers.append(nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            ))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        layers.append(nn.MaxPool2d(2,2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: Tensor):
        return self.block(x)

class VGG16(nn.Module):
    def __init__(self, num_of_classes: int):
        super().__init__()

        self.features = nn.Sequential(
            VGG16Block(3,64,2),
            VGG16Block(64,128,2),
            VGG16Block(128,256,3),
            VGG16Block(256,512,3),
            VGG16Block(512,512,3),
        )

        self.loss = nn.CrossEntropyLoss()

        self.classifier = nn.Sequential(
            nn.Linear(512*7*7,4096),
            nn.ReLU(True),
            nn.Linear(4096,4096),
            nn.ReLU(True),
            nn.Linear(4096,num_of_classes)
        )

    def forward(self, x: Tensor):
        x = self.features(x)
        x = x.view(x.size(0),-1)
        x = self.classifier(x)
        return x
    
    # def train(self,
    #       dataloader: torch.utils.data.DataLoader,
    #       optimizer: torch.optim.Optimizer,
    #       epochs: int = 5,
    #       device: torch.device = torch.device("cpu"),
    #       save_path: str = "checkpoints/vgg16_epoch{epoch}.pth"):

    #     self.to(device)
    #     self.train()

    #     loss_history = []

    #     for epoch in range(epochs):
    #         epoch_loss = 0.0
    #         for images, labels in dataloader:

    #             B, T, C, H, W = images.shape
    #             images = images.view(B*T, C, H, W)
    #             labels = labels.repeat_interleave(T)

    #             optimizer.zero_grad()
    #             outputs = self.forward(images)
    #             loss = self.loss(outputs, labels)
    #             loss.backward()
    #             optimizer.step()

    #             loss_history.append(loss.item())
    #             epoch_loss += loss.item()

    #         # Save checkpoint at the end of the epoch
    #         self.save(save_path.format(epoch=epoch+1), optimizer=optimizer, epoch=epoch+1, loss=epoch_loss/len(dataloader))
    #         print(f"Epoch {epoch+1}/{epochs} finished. Avg Loss: {epoch_loss/len(dataloader):.4f}. Checkpoint saved.")

    #     return loss_history
    
    def save(self, path, optimizer=None, epoch=None, loss=None):
        """
        Save model state_dict along with optional optimizer, epoch, loss.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {"model_state": self.state_dict()}
        if optimizer:
            state["optimizer_state"] = optimizer.state_dict()
        if epoch is not None:
            state["epoch"] = epoch
        if loss is not None:
            state["loss"] = loss

        torch.save(state, path)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path, input_dim, hidden_dim, output_dim, optimizer=None, map_location=None):
        """
        Load model and optionally optimizer. Returns model instance, epoch, loss
        """
        checkpoint = torch.load(path, map_location=map_location)
        model = cls(input_dim, hidden_dim, output_dim)
        model.load_state_dict(checkpoint["model_state"])
        if optimizer and "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        epoch = checkpoint.get("epoch", None)
        loss = checkpoint.get("loss", None)
        print(f"Model loaded from {path}")
        return model, epoch, loss


class VGG16BlockBNP(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs):
        super().__init__()

        layers = []

        for i in range(num_convs):
            layers.append(
                nn.Conv2d(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1
                )
            )
            layers.append(nn.BatchNorm2d(out_channels))  # ✅ BatchNorm
            layers.append(nn.ReLU(inplace=True))

        # ✅ Pooling after block
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)
    

class VGG16Plus(nn.Module):
    def __init__(self, num_of_classes: int):
        super().__init__()

        # Feature extractor
        self.features = nn.Sequential(
            VGG16BlockBNP(3, 64, 2),
            VGG16BlockBNP(64, 128, 2),
            VGG16BlockBNP(128, 256, 3),
            VGG16BlockBNP(256, 512, 3),
            VGG16BlockBNP(512, 512, 3),
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),        # ✅ Dropout

            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),        # ✅ Dropout

            nn.Linear(4096, num_of_classes)
        )

        self.loss = nn.CrossEntropyLoss()

    def forward(self, x: Tensor):
        x = self.features(x)

        # flatten
        x = torch.flatten(x, 1)

        x = self.classifier(x)
        return x