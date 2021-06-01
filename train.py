import pandas as pd
import numpy as np
import torch 
import torch.nn as nn 

def data_load():
    data = np.load("/home/mech-user/Desktop/3S/datascience/classification/dataset/train.npy",allow_pickle=True)
    X_train = torch.stack([torch.from_numpy(d) for d in data])
    target = np.load("/home/mech-user/Desktop/3S/datascience/classification/dataset/target.npy",allow_pickle=True)
    Y_train = torch.tensor(target)

    trainval_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    
    n_samples = len(trainval_dataset) 
    train_size = int(len(trainval_dataset) * 0.8) 
    val_size = n_samples - train_size 

    # shuffleしてから分割してくれる.
    train_dataset, val_dataset = torch.utils.data.random_split(trainval_dataset, [train_size, val_size])
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=32,shuffle=True,num_workers=2)
    valid_loader = torch.utils.data.DataLoader(val_dataset,batch_size=32,shuffle=True,num_workers=2)
    return train_loader,valid_loader

class LSTM(nn.Module):

    def __init__(self):
        super(LSTM,self).__init__()
        self.lstm = nn.LSTM(6,128,batch_first=True)
        self.fc1 = nn.Linear(128,6)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self,x):
    
        x,(h,c) = self.lstm(x)
        """print("x",x.shape)
        print("h",h.shape)
        print("c",c.shape)"""
        x = self.fc1(x[:,-1,:])
        #print(x.shape)
        x = self.softmax(x)
        return x

net = LSTM()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
#optimizer = torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    
def train(train_loader,valid_loader):
    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []

    nb_epoch = 100
    for epoch in range(nb_epoch):
        train_loss = 0
        train_acc = 0
        val_loss = 0
        val_acc = 0

        #train
        net.train()
        for i, (data, labels) in enumerate(train_loader):
        
            data = data.float()
            labels = labels.long()
            optimizer.zero_grad()
            outputs = net(data)
            loss = criterion(outputs, labels)
            train_loss += loss.item()
            train_acc += (outputs.max(1)[1] == labels).sum().item()
            loss.backward()
            optimizer.step()

        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_train_acc = train_acc / len(train_loader.dataset)

        train_loss_list.append(avg_train_loss)
        train_acc_list.append(avg_train_acc)
        print ('Epoch [{}/{}], loss: {loss:.4f} train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}' 
                    .format(epoch+1, nb_epoch, i+1, loss=avg_train_loss, train_loss=avg_train_loss, train_acc=avg_train_acc))
        #val
        net.eval()
        with torch.no_grad():
            for data, labels in valid_loader:
                data = data.float()
                labels = labels.long()
                outputs = net(data)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_acc += (outputs.max(1)[1] == labels).sum().item()
                avg_val_loss = val_loss / len(valid_loader.dataset)
                avg_val_acc = val_acc / len(valid_loader.dataset)

        print ('Epoch [{}/{}], loss: {loss:.4f} val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}' 
                    .format(epoch+1, nb_epoch, i+1, loss=avg_train_loss, val_loss=avg_val_loss, val_acc=avg_val_acc))
    
        val_loss_list.append(avg_val_loss)
        val_acc_list.append(avg_val_acc)

    model_path = 'model/lstm.pt'
    torch.save(net.state_dict(), model_path)
    

    import matplotlib.pyplot as plt

    plt.figure(figsize=(8,6))
    plt.plot(train_loss_list,label='train', lw=3, c='b')
    plt.plot(val_loss_list,label='test',lw=3,c = 'r')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('LSTM')
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.grid(lw=2)
    plt.legend(fontsize=14)
    plt.show()

    

    import matplotlib.pyplot as plt

    plt.figure(figsize=(8,6))
    plt.plot(train_acc_list,label='train', lw=3, c='b')
    plt.plot(val_acc_list,label='test',lw=3,c = 'r')
    plt.xlabel('Epoch')
    plt.ylabel('accuracy')
    plt.title('LSTM')
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.grid(lw=2)
    plt.legend(fontsize=14)
    plt.show()

if __name__ == "__main__":
    train_loader,valid_loader = data_load()
    train(train_loader,valid_loader)
    