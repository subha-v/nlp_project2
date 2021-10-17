import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128): #self refers to object we create
        super().__init__() 
        self.input_to_hidden = nn.Linear(input_dim+hidden_dim,hidden_dim)
        self.input_to_output = nn.Linear(input_dim+hidden_dim, output_dim) #both of these are linear functions
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.hidden_dim = hidden_dim

#dimensions are what we pass in as parameters
#this is how we define our teaching function

#defined fx, gx, hx in ^^^


    def forward(self, inp): #this sees how we combine them in our overall function
        hidden = torch.zeros(1,self.hidden_dim)
        num_letters = inp.size()[0] #this gives the input size
        for i in range(num_letters):
            combo = torch.cat([inp[i], hidden], dim=1)
            hidden = self.input_to_hidden(combo)
            out = self.input_to_output(combo)
            out = self.log_softmax(out)
        return out
        #this function defines how we go from a name to how we go to a value greek or korean


    #forward pass = plug in input and get output
    #the forward method is exactly that. describes how we manipulate our input to get our output