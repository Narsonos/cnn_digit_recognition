import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		#Conv2d(in_channels, out_features, kernel_size, ?)
		self.conv1 = nn.Conv2d(1,32,3,1) 
		self.conv2 = nn.Conv2d(32,64,3,1)

		#ensures that adjacent pixels are either 0 or active
		#with input probability
		self.dropout1 = nn.Dropout2d(0.25)
		self.dropout2 = nn.Dropout2d(0.5)

		self.fc1 = nn.Linear(9216,128)
		self.fc2 = nn.Linear(128,10)

	def forward(self, x):
		x = self.conv1(x)
		#using relu activation
		x = F.relu(x)
		x = self.conv2(x)
		x = F.relu(x)

		#maxpooling
		x = F.max_pool2d(x,2) #square frame 2
		x = self.dropout1(x)
		
		#Flattening input before passing to fully-connected layers
		x = torch.flatten(x,1)

		x = self.fc1(x)
		x = F.relu(x)
		x = self.dropout2(x)
		x = self.fc2(x)

		output = F.log_softmax(x,dim=1)
		return output




if __name__ == '__main__':

	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.5,),(0.5,)) 
		])
	
	train_ds = datasets.MNIST(root='./data',train=True,download=True,transform=transform)
	test_ds = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
	
	train_loader = DataLoader(dataset=train_ds, batch_size=128, shuffle=True)
	test_loader = DataLoader(dataset=test_ds, batch_size=128, shuffle=False)
	
	model = Net()
	criterion = nn.CrossEntropyLoss() #neg log likelihood loss
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
	
	ne = 5
	for e in range(ne):
		model.train() #change to training mode
		for imgs,labels in train_loader:
			optimizer.zero_grad()
			outs = model(imgs)
			loss = criterion(outs,labels)
			loss.backward() #backward propagation
			optimizer.step() #update weights and biases
	
			print(f'Epoch [{e+1}/{ne}], Loss:{loss.item():.4f}')
	
	model.eval() #change to evaluation mode
	correct, total = 0,0
	
	with torch.no_grad():
		for imgs,labels in test_loader:
			outputs = model(imgs)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted==labels).sum().item()
	
	print(f'Accuracy of the model on the test images: {100* correct / total:.2f}%')
	
	fname = 'model.pth'
	print(f'Saving model in {fname}')
	torch.save(model.state_dict(), fname)