import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
import sys
import os
import time
from dataset import CSVImageDataset

IMAGE_SIZE=(224,224) # h,w

Params = namedtuple('Params', ['lr','batch_size', 'epochs'])

def get_transform(train=True):
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation((-30,30)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
    ])
    return transform

def get_dataloaders(train_csv, val_csv, test_csv, root, params):
    train_set = CSVImageDataset(train_csv, sep=",", root=root, transforms=get_transform())
    val_set = CSVImageDataset(val_csv, sep=",", root=root, transforms=get_transform())
    test_set = CSVImageDataset(test_csv, sep=",", root=root,  transforms=get_transform())
    
    trainloader = torch.utils.data.DataLoader(train_set, batch_size = params.batch_size, shuffle=True, num_workers = 3)
    valloader = torch.utils.data.DataLoader(val_set, batch_size = params.batch_size, shuffle=False, num_workers = 3)
    testloader = torch.utils.data.DataLoader(test_set, batch_size = params.batch_size, shuffle = False, num_workers = 3)

    return trainloader, valloader, testloader


### Hyperparameters ###
def get_params(lr=0.0001, batch_size=32, epochs=10):
    params = Params(lr = lr, batch_size = batch_size, epochs = epochs)
    return params

def save_checkpoint(state, filename='chkpnt.pth.tar'):
  
    torch.save(state, filename)
    print("Saved %s" % filename)
    print()

### Train ###

def train(trainloader, valloader, device, optimizer, scheduler, net, criterion, n_epochs=10, ckpt_folder="checkpoint/"):
    net.train()
    train_start = time.time()
    for epoch in range(n_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # print statistics
            running_loss += loss.item()
            if i % 10 == 0:    
                running_loss = 0.0
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, i * len(inputs),
                len(trainloader.dataset),
                100. * i / len(trainloader), 
                loss.item())
            )
        print("Train duration: %f. Learning Rate: %f" % ( time.time() - train_start, optimizer.state_dict()["param_groups"][0]["lr"]))
#        print("LR: %f" %optimizer.state_dict()["param_groups"][0]["lr"])
        scheduler.step()
        # Validation
        validate(valloader, net, device)
        
        save_checkpoint(
            {
                'epoch':epoch,
                'state_dict': net.state_dict(),
                'optimiser' : optimizer.state_dict(),
            }, filename=os.path.join(ckpt_folder,("checkpoint_%d.pth.tar" % epoch))
        )
    print('Finished Training')
    return net

## Validation ##
@torch.no_grad()
def validate(valloader, net, device):
    net.eval()
    val_loss =0.0
    correct = 0

    for inputs, labels in valloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        
        # sum up batch loss
        val_loss += F.cross_entropy(outputs, labels, reduction='sum').item()
        
        # get the index of the max log-probability
        pred = outputs.argmax(dim=1, keepdim=True)
        correct += pred.eq(labels.view_as(pred)).sum().item()
    val_loss /= len(valloader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{}({:.0f}%)\n'.format(
             val_loss, correct, len(valloader.dataset),
             100. * correct / len(valloader.dataset))
      )






### Layers trainable ###
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


## Setup model ##

def get_model(num_classes, train_top_only = True, use_pretrained=True):
    model = torchvision.models.resnet50(pretrained = use_pretrained, progress = True)
    set_parameter_requires_grad(model, train_top_only)

    # Last layer settings
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes) 

    return model

## Eval ##
@torch.no_grad()
def eval_acc(net, testloader, device):
    '''
    Evaluation Function
    '''
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

### Main ###

def main(train_csv, val_csv, test_csv, root, params, ckpt_folder="checkpoint/", use_pretrained=False, top_only=False):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device) # prints cuda if on cuda

    # Data
    trainloader, valloader, testloader = get_dataloaders(train_csv, val_csv, test_csv, root, params)
    num_classes = trainloader.dataset.num_classes

    # Model
    net = get_model(num_classes, train_top_only=top_only, use_pretrained= use_pretrained).to(device)
    print(net)
    # Print model's state_dict
    print("Model's state_dict:")
    for param_tensor in net.state_dict():
        print(param_tensor, "\t", net.state_dict()[param_tensor].size())

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=params.lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=params.lr, max_lr=params.lr*9)
    
    # Train
    net = train(trainloader, valloader, device, optimizer, scheduler, net, criterion, params.epochs, ckpt_folder)
    
    # Evaluate
    eval_acc(net, testloader, device)

if __name__ == '__main__':
    try:
        train_csv = sys.argv[1].rstrip()
        test_csv = sys.argv[2].rstrip()
        val_csv = sys.argv[3].rstrip()
        root = sys.argv[4].rstrip()

        lr = float(sys.argv[5].rstrip())
        batch_size = int(sys.argv[6].rstrip())
        epochs = int(sys.argv[7].rstrip())
        
        ckpt_folder = sys.argv[8].rstrip()

        use_pretrained= sys.argv[9].rstrip()
        top_only = sys.argv[10].rstrip()

        if use_pretrained == "True":
            use_pretrained = True
        else:
            use_pretrained = False

        if top_only == "True":
            top_only = True
        else:
            top_only = False
            
        params = get_params(lr, batch_size, epochs)
        print(params)
    except Exception as e:
        print(e)
        print(sys.argv)
        print("Not enough parameters.")
        sys.exit()
    main(train_csv, val_csv, test_csv, root, params, ckpt_folder, use_pretrained, top_only)
