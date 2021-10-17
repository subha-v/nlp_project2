from model import *
from utils import *
from data import * #imports everything
import torch.nn as nn


class NameClassifier:
    def __init__(self, dataset_path, num_iters=1500):
        self.data = NamesDataset(dataset_path)
        self.num_iters = num_iters
        self.model = RNN(NUM_LETTERS,len(self.data.countries)) #fill up next time
        self.loss_func = nn.NLLLoss()
        self.learning_rate = 0.005 #how fase we go down the hill
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate) #instead of doing it in a gradient descent sort of way, we go down in a random way
        self.n_print=50
#instead of considering the whole mountain at once, just consider part of it
    def train_step(self, input_tensor, label_tensor):
        #label_tensor is the correct answer
        output = self.model(input_tensor) #model is like the students, and input tensor 
        # now we make a function to see how off the students answer is, and there are different ways to do this
        #for now its some function called loss_func
        loss = self.loss_func(output, label_tensor) #shows the error between output and correct answer
        #how do we make the model learn? using calculus
        #the optimizer uses the calculus
        self.optimizer.zero_grad() #this zeros all the gradiants
        loss.backward() #this computes the gradiants and its called back propagation which is explained on the paper
        self.optimizer.step() #stepping in the negative direction in the direction of the steepest gradient
        #this changes ws and bs in our model to make the error lower (move downhill)
        return output, loss.item() #loss is a tensor so we return the  t=tensor(1.) => t.item()=1

    
    def train(self): #think about teaching the model
        for i in range(self.num_iters):
            country, name, country_tensor, name_tensor = self.data.get_random_sample() #we initiated everything at once and make it a bundle object, which we can unpack
            # run 1 training loop step
            output, loss = self.train_step(name_tensor, country_tensor)

            if(i% self.n_print) == 0:
                prediction = self.data.output_to_country(output)
                print(f"Loss: {loss:.4f}, Prediction: {prediction}, Label: {country}") #ptings out this


    def predict(self, name, correct_label):
        input_tensor = name_to_tensor(name.lower()) #calls name to tensor on a given name
        output = self.model(input_tensor) #this gives us the two values
        country_pred = self.data.output_to_country(output) #prints out greek or korean
        print(f"Name:{name}, Prediction:{country_pred}, Label: {correct_label}")



                #prediction = #TO DO
                #get prediction and print it out 

if __name__ == '__main__':
    name_classifier = NameClassifier("data/names/*.txt")
    name_classifier.train()
    print()
    print("Predictions:")
    name_classifier.predict("Vadlamannati", "Indian")
    name_classifier.predict("Gampa", "Indian")
    name_classifier.predict("Osthamus", "Greek")
    name_classifier.predict("Cheng", "Korean")

    

   
    

#names the model has never seen


