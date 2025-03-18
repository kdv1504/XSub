import torch
from utils import config

class Trainer:
    def __init__(self, config, model, dataset, criterion, optimizer, 
                 train_data, test_data, lr_scheduler, epochs, save_period):
        self.config = config
        self.model = model
        self.dataset = dataset
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_data = train_data
        self.test_data = test_data
        self.lr_scheduler = lr_scheduler
        self.epochs = epochs
        self.save_period = save_period
    
    def train(self):
        self.model.train()
        device = self.config.device

        for epoch in range(1, self.epochs+1):
            total_correct, total_samples, cumu_loss = 0, 0, 0
            for batch_idx, (images, labels) in enumerate(self.train_data):
                images, labels = images.to(device), labels.to(device)
                total_samples += len(labels)

                self.optimizer.zero_grad()

                outputs = self.model(images)
                total_correct += torch.sum(torch.argmax(outputs, 1) == labels)

                loss = self.criterion(outputs, labels)
                cumu_loss += loss.item()
                loss.backward()

                self.optimizer.step()
            
            accuracy = total_correct/total_samples*100
            avg_loss = cumu_loss/total_samples
            print("Epoch: ", epoch, "Accuracy: ", accuracy, "Average loss: ", avg_loss)

            # Free up memory
            del images
            del labels
            del outputs

            if (epoch > 0) & (epoch % self.save_period == 0):
                torch.save(self.model.state_dict(), 
                           self.config.save_dir + 'models/model_' +self.dataset + '_' + str(epoch) + '.pt')
                print("Model saved.")
            self.lr_scheduler.step()
    
    def test(self):
        self.model.eval()
        device = self.config.device

        total_correct, total_samples = 0, 0
        for batch_idx, (images, labels) in enumerate(self.test_data):
            images, labels = images.to(device), labels.to(device)
            total_samples += len(labels)

            outputs = self.model(images)
            total_correct += torch.sum(torch.argmax(outputs, 1) == labels)

            # Free up memory
            del images
            del labels
            del outputs
        
        accuracy = total_correct/total_samples*100
        print("Test accuracy: ", accuracy)