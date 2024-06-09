import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import VOCSegmentation

# Define the Segmentation Model with ResNet18 Backbone
class ResNetUNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        base_model = torchvision.models.resnet18(pretrained=True)
        self.base_layers = list(base_model.children())
        self.layer0 = nn.Sequential(*self.base_layers[:3])
        self.layer1 = nn.Sequential(*self.base_layers[3:5])
        self.layer2 = self.base_layers[5]
        self.layer3 = self.base_layers[6]
        self.layer4 = self.base_layers[7]

        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.upconv1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)

        self.conv_up3 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv_up2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv_up1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.conv_last = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        upconv4 = self.upconv4(layer4)
        upconv4 = torch.cat([upconv4, layer3], dim=1)
        upconv4 = self.conv_up3(upconv4)

        upconv3 = self.upconv3(upconv4)
        upconv3 = torch.cat([upconv3, layer2], dim=1)
        upconv3 = self.conv_up2(upconv3)

        upconv2 = self.upconv2(upconv3)
        upconv2 = torch.cat([upconv2, layer1], dim=1)
        upconv2 = self.conv_up1(upconv2)

        upconv1 = self.upconv1(upconv2)
        upconv1 = torch.cat([upconv1, layer0], dim=1)

        out = self.conv_last(upconv1)
        return out

# Define Training and Validation Transforms
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Download Pascal VOC Dataset
train_dataset = VOCSegmentation(root="./data", year="2012", image_set='train', download=True, transform=transform, target_transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Training the Model
def train_model(model, dataloader, criterion, optimizer, num_epochs=25):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, masks in dataloader:
            images, masks = images.cuda(), masks.cuda()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"Epoch {epoch}/{num_epochs - 1}, Loss: {epoch_loss:.4f}")

    return model

# Initialize and Train the Model
model = ResNetUNet(n_classes=1).cuda()
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
model = train_model(model, train_loader, criterion, optimizer, num_epochs=25)

# Save the Trained Model
torch.save(model.state_dict(), 'resnet_unet_model.pth')

# Inference
def predict(model, image):
    model.eval()
    with torch.no_grad():
        image = image.unsqueeze(0).cuda()
        output = model(image)
        return torch.sigmoid(output).squeeze().cpu().numpy()

# Example Inference
test_image = Image.open('path/to/test/image.png').convert("RGB")
test_image = transform(test_image)
prediction = predict(model, test_image)