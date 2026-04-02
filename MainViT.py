import torch
from DeepLearning.ViTModel import ViTTrain
from Utils.DataUtils.PyTorchDataLoader import create_dataloaders

batch_size = 16
train_loader, val_loader, test_loader = create_dataloaders(batch_size=batch_size)

num_classes = 5
trained_model = ViTTrain(train_loader, test_loader, val_loader, num_classes, epochs=50, show=True)

torch.save(trained_model.state_dict(), "motionClassificationModel_ViT.pth")
print("Model saved to motionClassificationModel_ViT.pth")
