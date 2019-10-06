import torch
import matplotlib.pyplot as plt

def activation_results():
    x = torch.range(-5.,5.,0.1)
    sigmoid = torch.sigmoid(x)
    tanh = torch.tanh(x)
    relu_activation = torch.nn.ReLU()
    relu = relu_activation(x)
    prelu_activation = torch.nn.PReLU(num_parameters=1)
    prelu = prelu_activation(x)

    return x, [sigmoid, tanh, relu, prelu]

def softmax_activation():
    softmax = torch.nn.Softmax(dim=1)
    x_input = torch.randn(1,3)
    y_output = softmax(x_input)
    print(x_input)
    print(y_output)
    print(torch.sum(y_output,dim=1))

x, activations = activation_results()
# plt.figure(22)
for index, i in enumerate(activations):
    plt.subplot(221+index)
    # plt.subplot((int(index/2)+1) * 10 + index % 2+1)
    plt.plot(x.numpy(),i.detach().numpy())

plt.savefig('activations.png')
softmax_activation()
