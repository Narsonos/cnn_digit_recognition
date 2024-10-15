import numpy as np 

#To watch: https://www.youtube.com/watch?v=pauPCy_s0Ok&ab_channel=TheIndependentCode
#For Dense and activation classses, just copied code, haven't watched yet
class Dense():
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)

    def forward(self,input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias 

    def backward(self, output_grad, lr):
        #print(output_grad.shape, self.input.T.shape)
        weights_grad = np.dot(output_grad, self.input.T)
        self.weights -= lr * weights_grad
        self.bias -= lr * output_grad
        return np.dot(self.weights.T, output_grad)

class Activation():
    #activation is an activation function
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input 
        return self.activation(self.input) 

    def backward(self, output_grad, lr):
        #print(output_grad.shape, self.activation_prime(self.input).shape)
        return np.multiply(output_grad, self.activation_prime(self.input))




class ConvLayer():
    def __init__(self, input_shape, kernel_size, depth):
        input_depth,input_height,input_width = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (depth,input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)

        self.kernels = np.random.randn(*self.kernels_shape)
        self.biases = np.random.randn(*self.output_shape)
        print(self.biases.shape)

    def forward(self, input):
        self.input = input
        self.output = np.copy(self.biases)
        for i in range(self.depth):
            for j in range(self.input_depth):
                #print("input[j] shape:", self.input.shape,self.input[j].shape)
                #print("kernels[i,j] shape:",self.kernels[i,j].shape)
                #print("output[i] shape:", self.output[i].shape)
                self.output[i] += self.cross_correlate(self.input[j],self.kernels[i,j],mode='valid')
        return self.output

    def backward(self,output_grad, lr):
        kernels_grad = np.zeros(self.kernels_shape)
        input_grad = np.zeros(self.input_shape)

        for i in range(self.depth):
            for j in range(self.input_depth):
                kernels_grad[i,j] = self.cross_correlate(self.input[j],output_grad[i],mode='valid')
                input_grad[j] += self.convolve(output_grad[i], self.kernels[i,j],mode="full")
        self.kernels -= lr * kernels_grad
        self.biases -= lr * output_grad
        return input_grad


    #PURE PYTHON REALISATION
    #mode: valid only
    #for convolution -- rotate the kernel_block
    #TODO: Add numpy realisation to simplify + speed up some of the ops
    def cross_correlate(self, input_block, kernel_block, mode='valid', realisation='python'):

        #input_block dims = kernel_block dims
        #print("Input shape:", input_block.shape)
        #print("Kernel shape:", kernel_block.shape)
        input_width,input_height = (input_block.shape)
        kernel_width,kernel_height = (kernel_block.shape)
        output_shape = (input_height - kernel_height + 1, input_width - kernel_width + 1)
        output_width, output_height = output_shape
        output = []

        if realisation == 'python':
            if mode == 'valid':
                #range_x =  range(self.output_shape[2])
                #range_y = range(self.output_shape[1])
                range_x = range(output_width)
                range_y = range(output_height)
            elif mode == 'full':
                #range_x = range(1-self.kernels_shape[2],self.input_shape[2])
                #range_y = range(1-self.kernels_shape[2],self.input_shape[1])
                range_x = range(1-kernel_width,input_width)
                range_y = range(1-kernel_height,input_height)
            for shift_x in range_x:
                temp_row = []
                #print(output)
                for shift_y in range_y:
                    temp = 0
                    #print('shifts:',shift_x,shift_y)
                    for i in range(shift_x, shift_x+kernel_width): #loops over x within input subblock
                        for j in range(shift_y, shift_y + kernel_height): #loops over x within input subblock
                            #print(i,j)
                            try:
                                temp += input_block[i][j] * kernel_block[i-shift_x][j-shift_y]
                            except IndexError:
                                #print(f"Can't get {i},{j}, making it zero")
                                continue
                    #print(temp)
                    temp_row.append(temp)
                output.append(temp_row)

        

        return output

    def convolve(self,input_block, kernel_block, mode='valid', realisation="python"):
        rotated_kernel = np.flip(kernel_block)
        output = self.cross_correlate(input_block, rotated_kernel, mode=mode,realisation=realisation)
        return output





class ReshapeLayer():
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input):
        #print(f'[reshape]in_shape:{input.shape}')
        #print(f'[reshape]out_shape:{self.output_shape}')
        return np.reshape(input, self.output_shape)

    def backward(self, output_grad, lr):
        return np.reshape(output_grad, self.input_shape)


#Loss function
class BinaryCrossEntropy():
    @staticmethod
    def entropy(y_true, y_pred):
        return -np.mean(y_true * np.log(y_pred) + (1-y_true) * np.log(1-y_pred))
    @staticmethod
    def entropy_prime(y_true, y_pred):
        return ((1 - y_true) / (1-y_pred) - y_true / y_pred) / np.size(y_true)

class CategoricalCrossEntropy():
    @staticmethod
    def entropy(y_true, y_pred):
        # Clip predictions to prevent log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    @staticmethod
    def entropy_prime(y_true, y_pred):
        # Clip predictions to prevent division by zero
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return (y_pred - y_true) / y_pred.shape[0]


class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1+np.exp(-x))
    
        def sigmoid_prime(x):
            return sigmoid(x) * (1-sigmoid(x))

        super().__init__(sigmoid, sigmoid_prime)

class ReLU(Activation):
    def __init__(self):
        def relu(x):
            return np.maximum(0, x)

        def relu_prime(x):
            return np.where(x > 0, 1, 0)

        super().__init__(relu, relu_prime)



class Softmax():
    def forward(self,input):
        tmp = np.exp(input)
        self.output = tmp / np.sum(tmp)
        return self.output

    def backward(self, output_grad, lr):
        n = np.size(self.output)
        tmp = np.tile(self.output, n)
        return np.dot(tmp * (np.identity(n) - np.transpose(tmp)), output_grad)




if __name__ == "__main__":
    
    ###some dev tests
    
    #cl = ConvLayer((1,3,3),2,1)
    #for x in cl.kernels:
    #   for kernel_row in x:
    #       print(kernel_row)
    #input_block = [[
    #[1,6,2],
    #[5,3,1],
    #[7,0,4]
    #]]
    #rs = cl.cross_correlate(input_block[0], cl.kernels[0][0], mode='full')
    ##rs = cl.forward(input_block)
    #print("output:")
    #for row in rs:
    #   print(row)
    #print("")
    import torch.nn.functional as F 
    import torch.nn as tnn

    nn = [
        ConvLayer((1,28,28), 3, 5),
        Sigmoid(),
        #ConvLayer((4,26,26), 3, 8),
        #Sigmoid(),
        ReshapeLayer((5,26,26),(5*26*26,1)),
        Dense(5*26*26,100),
        Sigmoid(),
        Dense(100,10),
        Softmax()
    ]

    #nn = [
    #    ConvLayer((1,28,28), 3, 5),
    #    F.relu(),
    #    #Sigmoid(),
    #    ConvLayer((1,28,28), 3, 5),
    #    ReshapeLayer((5,26,26),(5*26*26,1)),
    #    Dense(5*26*26,100),
    #    Sigmoid(),
    #    Dense(100,10),
    #    Softmax()
    #]    

    epochs = 20
    lr = 0.1

    from keras.datasets import mnist
    from keras import utils as np_utils
    from keras.losses import CategoricalCrossentropy

    entropy_obj = CategoricalCrossEntropy()


    def preprocess_data(x,y,limit):
        zero_index = np.where(y==0)[0][:limit]
        one_index = np.where(y==1)[0][:limit]
        all_indices = np.hstack((zero_index, one_index))
        all_indices = np.random.permutation(all_indices)
        x,y = x[all_indices], y[all_indices]
        x = x.reshape(len(x),1,28,28)
        x = x.astype('float32')/255
        y= np_utils.to_categorical(y)
        y=y.reshape(len(y),2,1)
        return x,y


    def preprocess_data(x, y, limit=None):
        if limit is not None:
            # If a limit is provided, select limited samples for each class
            all_indices = []
            for digit in range(10):  # For digits 0-9
                digit_indices = np.where(y == digit)[0][:limit]  # Get indices for the current digit
                all_indices.extend(digit_indices)  # Add to the list
            
            all_indices = np.array(all_indices)  # Convert to a NumPy array
            np.random.shuffle(all_indices)  # Shuffle the indices
        else:
            all_indices = np.arange(len(y))  # Use all indices if no limit is specified
    
        # Subset the data
        x, y = x[all_indices], y[all_indices]
    
        # Reshape x for CNN input
        x = x.reshape(len(x), 1, 28, 28)  # Reshape to (num_samples, channels, height, width)
        x = x.astype('float32') / 255  # Normalize to [0, 1]
    
        # One-hot encode the labels for multi-class classification
        y = np_utils.to_categorical(y, num_classes=10)  # Change to 10 classes for digits 0-9
        y = y.reshape(len(y),10,1)

    
        return x, y




    (x_train,y_train),(x_test,y_test) = mnist.load_data()
    x_train, y_train = preprocess_data(x_train, y_train, 100)
    x_test, y_test = preprocess_data(x_test, y_test, 100)
    #print(x_train)
    i = 0 
    for e in range(epochs):
        error = 0
        for image, label in zip(x_train,y_train): 
            i += 1
            output = image 
            #print(f'Epoch: {e}, sample: {i}')
            for layer in nn:
                output = layer.forward(output)
                #print(f'{layer.__class__.__name__} output shape: {output.shape}')

            

            loss = entropy_obj.entropy(label,output)

            error += loss

            grad = entropy_obj.entropy_prime(label,output)

            #print('---BACKWARD PASS---')
            for layer in reversed(nn):
                #print(f'{layer.__class__.__name__} input shape: {grad.shape}')
                grad = layer.backward(grad, lr)
                #print(f'{layer.__class__.__name__} output shape: {grad.shape}')

        error /= len(x_train)
        print(f"{e+1}/{epochs}, error = {error}")

    correct, total = 0,0
    for x,y in zip(x_test,y_test):
        output = x
        for layer in nn:
            output = layer.forward(output)
        true, predicted = np.argmax(y), np.argmax(output)
        print(true,predicted)
        total += 1
        correct += (predicted==true).sum().item()
    
    print(f'Accuracy of the model on the test images: {100* correct / total:.2f}%')

    import pickle
    
    # Assuming `model` is your neural network instance
    with open('model_np.pkl', 'wb') as file:
        pickle.dump(nn, file)