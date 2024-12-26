from model import Model
from dataset import dataset, normalize, split_to_batches
from tqdm import tqdm
from torch import nn, optim
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_one_epoch(model, dataset, optimizer, loss_fn):
    running_loss = 0
    for data in dataset:
        inputs, labels = data
        inputs = torch.tensor(inputs, dtype=torch.float32, device=device)
        labels = torch.tensor(labels, dtype=torch.float32, device=device)
        
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    return running_loss

def eval(model, dataset):
    inputs, labels = dataset
    inputs = torch.tensor(inputs, dtype=torch.float32, device=device)
    labels = torch.tensor(labels, dtype=torch.float32, device=device)

    outputs = model(inputs)
    outputs = torch.argmax(outputs, dim=1).cpu().numpy()

    labels = torch.argmax(labels, dim=1).cpu().numpy()

    return confusion_matrix(outputs, labels)

if __name__ == "__main__":
    BATCH_SIZE = 5
    EPOCH = 1000

    model = Model().to(device)
    loss_fn = nn.CrossEntropyLoss()
    running_loss = 0

    train_dataset, test_dataset = dataset()
    train_features, train_labels = train_dataset
    train_features, maximum, minimum = normalize(train_features)

    train_dataset = (train_features, train_labels)
    train_dataset = split_to_batches(train_dataset, BATCH_SIZE)
    
    test_features, test_labels = test_dataset
    test_features, maximum, minimum = normalize(test_features, maximum, minimum)
    
    test_dataset = (test_features, test_labels)
    
    losses = []
    for i in tqdm(range(EPOCH)):
        optimizer = optim.SGD(model.parameters(), 0.01, 0.003)
        running_loss += train_one_epoch(model, train_dataset, optimizer, loss_fn)
        losses.append(running_loss)
    
    plt.plot(losses)
    plt.show()
    
    cm = eval(model, test_dataset)
    classes = ("Good", "Moderate", "Poor", "Hazardous")
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels= classes)
    cm_display.plot()
    plt.show()
