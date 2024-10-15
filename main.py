import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from model import Net
import tkinter as tk
from PIL import Image, ImageDraw
import PIL.ImageOps





class App():
    def __init__(self):
        self.root = tk.Tk()
        self.root.geometry('300x400')

        self.cnv_x_size = 112
        self.cnv_y_size = 112

        self.cnv = tk.Canvas(self.root,highlightthickness=1, highlightbackground="black", height=112,width=112, bd=3, bg='white')
        self.cnv.place(x=(300-self.cnv_x_size)//2,y=50)

        self.b1 = "up"
        self.xold = None
        self.yold = None 
        self.image = None

        self.image=Image.new("RGB",(self.cnv_x_size,self.cnv_y_size),(255,255,255))
        self.draw= ImageDraw.Draw(self.image)

        self.cnv.bind("<Motion>", self.motion)
        self.cnv.bind("<ButtonPress-1>", self.b1down)
        self.cnv.bind("<ButtonRelease-1>", self.b1up)
        self.cnv.bind("<Button-3>",self.clear)
        self.root.bind_all("<Return>",self.check)
        self.root.bind_all("<space>",self.save)

    def save(self,event):
        filename = "my_drawing.jpg"
        print(f'saving img as {filename}')
        self.image = self.image.resize((28, 28), 1)
        self.image = self.image.convert('L')
        self.image.save(filename)

    def clear(self, *args,**kwargs):
        self.cnv.delete("all")
        self.image=Image.new("RGB",(self.cnv_x_size,self.cnv_y_size),(255,255,255))
        self.draw=ImageDraw.Draw(self.image)

    def b1down(self,event):
        self.b1 = "down"

    def b1up(self,event):
        self.b1 = "up"
        self.xold = None
        self.yold = None


    def motion(self,event):
        if self.b1 == "down":
            if self.xold is not None and self.yold is not None:
                event.widget.create_line(self.xold,self.yold,event.x,event.y,smooth='true',width=4,fill='blue')
                self.draw.line(((self.xold,self.yold),(event.x,event.y)),(0,128,0),width=4)

        self.xold = event.x
        self.yold = event.y

    def check(self,event):
        self.image = PIL.ImageOps.invert(self.image)
        self.image = self.image.convert('L')
        image_tensor = NN.img_transform(self.image)
        image_tensor = image_tensor.unsqueeze(0)
        output = NN(image_tensor)
        print(F.softmax(output,dim=1))
        predicted_class = torch.argmax(output, dim=1)
        print(predicted_class.item())
        print([f'{x[0]} = {x[1]:.2f}' for x in enumerate(F.softmax(output,dim=1)[0])])
        self.clear()



NN = Net()
NN.load_state_dict(torch.load('model.pth'))
NN.eval()
NN.img_transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,),(0.5,)) 
        ])

app = App()
app.root.mainloop()