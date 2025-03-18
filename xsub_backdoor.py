'''Importing packages'''
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import shap
import time
import random

'''Set up CUDA memory'''
use_cuda = True
print("CUDA: ", torch.cuda.is_available())
device = torch.cuda.device("cuda" if use_cuda else "cpu")
print("Using", torch.cuda.device_count(), "GPUs")

torch.manual_seed(1512)

def ld_cifar10(batch_size, num_workers):
    """Load training test data."""

    transform_train = transforms.Compose([transforms.ToTensor()])
    # transforms.RandomCrop(32, padding=4),
    # transforms.RandomHorizontalFlip(),
    # transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
    cifar_trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(cifar_trainset, batch_size=int(batch_size),shuffle=True, num_workers=num_workers)

    i = 0
    for data, target in train_loader:
        if i == 0:
            train_data = data
            train_labels = target
        else:
            train_data = torch.cat((train_data, data))
            train_labels = torch.cat((train_labels, target))
        i+=1
    print("Train data shape: ", train_data.shape)
    print("Labels shape: ", train_labels.shape)

    return train_data, train_labels

batch_size = 1000
num_workers = 2
train_data, train_labels = ld_cifar10(batch_size, num_workers)

cwd = os.getcwd()

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=48, kernel_size=(3,3), padding=(1,1))
        self.conv2 = nn.Conv2d(in_channels=48, out_channels=96, kernel_size=(3,3), padding=(1,1))
        self.conv3 = nn.Conv2d(in_channels=96, out_channels=192, kernel_size=(3,3), padding=(1,1))
        self.conv4 = nn.Conv2d(in_channels=192, out_channels=256, kernel_size=(3,3), padding=(1,1))
        self.pool1 = nn.MaxPool2d(2,2)
        self.pool2 = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(in_features=8*8*256, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=64)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.dropout3 = nn.Dropout(0.25)
        self.fc3 = nn.Linear(in_features=64, out_features=10)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        self.relu5 = nn.ReLU()
        self.relu6 = nn.ReLU()
        # inplace=True


    def forward(self, x):
        #print(x.shape)
        #x = F.relu(self.conv1(x)) #32*32*48
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x)) #32*32*96
        x = self.pool1(x) #16*16*96
        x = self.dropout1(x)
        x = self.relu3(self.conv3(x)) #16*16*192
        x = self.relu4(self.conv4(x)) #16*16*256
        x = self.pool2(x) # 8*8*256
        x = self.dropout2(x)
        x = x.view(-1, 8*8*256) # reshape x
        x = self.relu5(self.fc1(x))
        x = self.relu6(self.fc2(x))
        x = self.dropout3(x)
        x = self.fc3(x)
        return x

print('Loading model...')
model = ConvNet()
model.cuda()
# model.load_state_dict(torch.load(os.path.join(cwd,'cifar',"model_with_epoch" + str(50) + ".pth")))

'''initialize explainer for later use'''
e = shap.DeepExplainer(model, torch.cuda.FloatTensor(train_data[:100].cuda()))
print("Successfully initialized explainer.")


'''Find top k pixels/positions with highest SHAP values'''
def find_topk_shap(shp, topk):
    '''
    Find top-k positions that have the highest SHAP values
    '''
    (rows, cols) = shp.shape
    max_shap = 0
    for j in range(rows):
        for k in range(cols):
            curr_max = shp[j,k]
            if curr_max > max_shap:
                max_shap = curr_max
                topk_id = [j,k]
    return topk_id


''' Main perturbation function'''
def pixel_swap(true_img, target_img, true_shp, target_shp, topk_true, topk_target, alpha, beta, strategy):
    a = find_topk_shap(true_shp, topk_true)
    b = find_topk_shap(target_shp, topk_target)
    to_return = torch.clone(true_img)

    for j in range(true_img.shape[0]):
        subt = -alpha * to_return[j, a[0], a[1]]
        add = beta * target_img[j, b[0], b[1]]
        to_return[j, a[0], a[1]] = to_return[j, a[0], a[1]] + subt + add
        if strategy == 'clipping':
            if to_return[j, a[0], a[1]] > 1:
                to_return[j, a[0], a[1]] = 1
            elif to_return[j, a[0], a[1]] < 0:
                to_return[j, a[0], a[1]] = 0

    return to_return

'''Calculate perturbation rate'''
def pert_change(orig, pert):
    '''
    Parameters:
    orig = original image
    pert = perturbed image
    '''

    pert_rate = round(torch.norm((pert.cpu()- orig.cpu()).cuda(), 2).item(),2)

    return pert_rate


'''find the best target image'''
def find_golden_img(data, labels):
    golden_idx = [0] * 10
    max_shap_list = [0] * 10
    for i in range(data.shape[0]):
        data_instance = data[i].reshape(1,3,32,32)
        shap_instance = torch.sum(torch.tensor(e.shap_values(data_instance)), 1) # final output shape: (1,32,32,10)
        curr_max = torch.max(shap_instance[:, :, :, labels[i]])
        # print("Current max: ", curr_max, "at label: ", labels[i])
        if max_shap_list[labels[i]] < curr_max:
            max_shap_list[labels[i]] = curr_max
            golden_idx[labels[i]] = i
    return golden_idx, max_shap_list


# golden_idx, max_shap_list = find_golden_img(train_data, train_labels) # takes about 3 hours
golden_idx = [2258, 48886, 34305, 29157, 31402, 27535, 33095, 41728, 44217, 10555]

'''Poisoning images'''
'''Set up hyperparameters first'''
poison_perc = 0.05
alpha = 300
beta = 300
k_true = 1
k_target = 1
strategy = 'no clip'

n_poison = int(train_data.shape[0] * poison_perc)
poison_idx = np.random.permutation(train_data.shape[0])[:n_poison]

start_time = time.time()
for i in poison_idx:
    '''obtaining labels'''
    true_label = train_labels[i]
    target_label = random.randint(0, 9)
    while target_label == true_label:
        target_label = random.randint(0, 9)

    '''obtaining images'''
    true_img = train_data[i]
    target_img = train_data[golden_idx[target_label]]

    '''obtaining SHAP values'''
    true_shp = torch.sum(torch.tensor(e.shap_values(true_img.reshape(1,3,32,32))), 2) # final output shape: (10,1,32,32)
    true_shp = true_shp[true_label, 0, :, :]
    target_shp = torch.sum(torch.tensor(e.shap_values(target_img.reshape(1,3,32,32))), 2) # final output shape: (10,1,32,32)
    target_shp = target_shp[target_label, 0, :, :]

    '''poison image'''
    im = pixel_swap(true_img, target_img, true_shp, target_shp, 1, 1, alpha, beta, strategy)
    im = im.reshape(1,3,32,32)

    train_data = torch.cat((train_data, torch.tensor(im))) # add poisoned image to training set
    train_labels = torch.cat((train_labels, torch.tensor([true_label]))) # add ground-truth label to training set

poisoning_time = (time.time() - start_time)/60
print('Time elapsed for poisoning: %.2f min' % (poisoning_time)) # takes about ? hours


torch.save(train_data, "train_data.pt")
torch.save(train_labels, "train_labels.pt")

# train_data = torch.load("train_data.pt")
# train_labels = torch.load("train_labels.pt")

print("New training data shape:", train_data.shape)
print("New training labels shape:", train_labels.shape)
print("Unique ground-truth training labels: ", torch.unique(train_labels, return_counts=True))

# Loss function:
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200, 300, 400], gamma=0.1)
#optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Training loop
start_time = time.time()
model.train()
epochs = 500
for epoch in range(epochs):  # loop over the dataset multiple times

    total_loss = 0.0
    for i in range(1, int(train_data.shape[0]/100+1)):
        # get the inputs
        inputs = train_data[(i-1)*100:i*100]
        labels = train_labels[(i-1)*100:i*100]

        # transform the input data
        transform_inputs = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
        inputs = transform_inputs(inputs)
        #inputs = inputs.reshape(1,3,32,32)
        inputs = torch.cuda.FloatTensor(inputs.cuda())

        # transform labels
        labels = labels.cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # # sanity check
        # print("Input shape:", inputs.shape)
        # print("Label: ", labels.shape)

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

scheduler.step()

print('Finished Training.')
training_time = (time.time() - start_time)/60
print('Time elapsed for training: %.2f min' % (training_time)) # takes about ? hours
PATH = './cifar_net.pth'
torch.save(model.state_dict(), PATH)


'''Testing'''
'''Load collated test data'''
test_data = torch.tensor(np.load("original.npy"))
test_labels = torch.tensor(np.load("labels.npy"))

# Load collated SHAP values
i = 0
for filename in os.listdir('cifar/'):
    if filename[0] == "s":
        if i == 0:
            shp = np.load('cifar/'+ filename)
        else:
            shp = np.append(shp, np.load('cifar/'+ filename), axis = 0)
        i += 1
for i in range(shp.shape[2]):
    if i == 0:
        agg_shp = shp[:,:,i]
    else:
        agg_shp += shp[:,:,i]
print("Aggregate SHAP shape:", agg_shp.shape)

# model accuracy
correct = 0
for i in range(test_data.shape[0]):
    im = test_data[i]
    im = torch.cuda.FloatTensor(torch.tensor(im).cuda())
    transform_im = transforms.Compose([
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    im = transform_im(im)
    im = im.reshape(1,3,32,32)
    _, im_pred = torch.max(model(im).data, 1)

    # pred = predict(model, torch.cuda.FloatTensor(instance.cuda()))
    if im_pred.item() == test_labels[i]:
        correct +=1
clean_accuracy = correct/test_data.shape[0]*100
print("Model accuracy:", clean_accuracy, "%")

# adversarial attack success rate - takes about 2 hours
start_time = time.time()
t_success, u_success, pert = 0, 0, 0
adv_golden_idx = [5294, 7855, 7340, 4056, 5258, 1745, 6865, 5635, 1812, 348]
golden_shp = torch.load("golden_shp.pt")
print("Golden SHAP shape: ", golden_shp.shape)

for i in range(test_data.shape[0]):
    # obtaining labels
    true_label = test_labels[i]
    target_label = random.randint(0, 9)
    while target_label == true_label:
        target_label = random.randint(0, 9)

    # obtaining images
    true_img = test_data[i]
    target_img = test_data[adv_golden_idx[target_label]]

    # obtaining SHAP values
    # true_shp = torch.sum(torch.tensor(e.shap_values(true_img.reshape(1,3,32,32))), 2) # final output shape: (10,1,32,32)
    # true_shp = true_shp[true_label, 0, :, :]
    true_shp = agg_shp[i][true_label]
    # target_shp = torch.sum(torch.tensor(e.shap_values(target_img.reshape(1,3,32,32))), 2) # final output shape: (10,1,32,32)
    # target_shp = target_shp[target_label, 0, :, :]
    target_shp = golden_shp[target_label][target_label]

    # poison image + get distance from original image
    im = pixel_swap(true_img, target_img, true_shp, target_shp, 1, 1, alpha, beta, strategy)
    pert += pert_change(true_img, im)

    # make prediction
    im = im.reshape(1,3,32,32)
    transform_im = transforms.Compose([transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    im = transform_im(im)
    _, im_pred = torch.max(model(im.cuda()).data, 1)
    if im_pred == target_label:
        t_success +=1
        u_success +=1
    elif im_pred != true_label:
        u_success +=1
t_return = t_success / test_data.shape[0]
u_return = u_success / test_data.shape[0]
pert = pert / test_data.shape[0]
adv_time = (time.time() - start_time)/60
print('Time elapsed for attack: %.2f min' % (adv_time))
print(f'Success rate = {round(t_return, 4)} and {round(u_return, 4)}; Average L2-norm perturbation: {round(pert, 4)}\n')
with open('backdoor_outputs.txt', 'w') as f:
    f.write(f"Poisoning time: {poisoning_time}\n")
    f.write(f"Training time: {training_time}\n")
    f.write(f"Adversarial attack time: {adv_time}\n")
    f.write(f"Clean accuracy: {clean_accuracy}\n")
    f.write(f"Advesarial attack success rate: {round(t_return, 4)} and {round(u_return, 4)} \n")
    f.write(f"Average L2-norm perturbation: {round(pert, 4)}\n")
