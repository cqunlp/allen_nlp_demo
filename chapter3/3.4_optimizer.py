import torch.nn as nn
import torch.optim as optim
from perceptron import Perceptron
input_dim = 2
lr = 0.01

percetron = Perceptron(input_dim = input_dim)
bce_loss = nn.BCELoss()

optimizer = optim.Adam(params = percetron.parameters(),lr = lr)

# load data
def get_toy_data(batch_size):

    return [],[]


# supervised training loop

n_epochs = 10
n_batchs = 10
batch_size = 1
# each epoch is a complete pass over the training data
for i in range(n_epochs):
    # the inner loop is over the batches in the dataset
    for batch in range(n_batchs):
        # Step 0: Get the data
        x_data, y_data = get_toy_data(batch_size)

        # Step 1: Clear the gradients
        percetron.zero_grad()

        # Step 2: Compute the forward pass of the model
        y_pred = percetron(x_data)

        # Step 3: Compute the loss value that we wish to optimize
        loss = bce_loss(y_pred, y_data)

        # Step 4: Propagate the loss signal backward
        loss.backward()

        # Step 5: Trigger the optimizer to perform one update
        optimizer.step()


