import torch
import torch.nn as nn
from logger import setup_logs
from torch.autograd import Variable
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from load_data import loadDataSet
from timeit import default_timer as timer
import os
import time
import logging



trainfolder = "fold1_train"
testfolder = "fold1_test"

'''
# Run name
run_name = time.strftime("%Y-%m-%d_%H%M-") + "CNN"

# Snapshot
save_dir = './snapshots'

if __name__ == "__main__":
    global_timer = timer()
    logger = setup_logs(save_dir, run_name)
'''


# Directory
directory = os.path.dirname(os.path.realpath(__file__))
audio_path = os.path.join(directory,"audio")
mfcc_path = os.path.join(directory,"mfcc_output")
crop_path = os.path.join(directory,"crop")
train_path = os.path.join(directory,"crop",trainfolder,"img/")
test_path = os.path.join(directory,"crop",testfolder,"img/")


# Hyper Parameters
EPOCH = 1               # Training the dataset n times
BATCH_SIZE = 1
LR = 0.001              # Learning rate


X_train = loadDataSet(os.path.join(directory,"fold1_train.csv"),train_path,ToTensor())
train_loader = DataLoader(X_train, shuffle=True)

X_test = loadDataSet(os.path.join(directory,"fold1_evaluate.csv"),test_path,ToTensor())
test_loader = DataLoader(X_train, shuffle=True)

# CNN Model (2 Conv Layer)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(4, 16, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)) # output shape (16,152,362)
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)) # output shape (32,76,181)
        self.fc = nn.Linear(76 * 181 * 32, 15)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

cnn = CNN()
print(cnn)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)

# Train the Model
for epoch in range(EPOCH):
    for i, (img_name,img_label) in enumerate(train_loader):
        images = Variable(img_name)
        labels = Variable(img_label)
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = cnn(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                  % (epoch + 1, EPOCH, i + 1, len(X_train) // BATCH_SIZE, loss.data[0]))

# Test the Model
cnn.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
correct = 0
total = 0
for images, labels in test_loader:
    images = Variable(images)
    outputs = cnn(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Test Accuracy of the model on the 880 test images: %d %%' % (100 * correct / total))

# Save the Trained Model
torch.save(cnn.state_dict(), 'cnn.pkl')




