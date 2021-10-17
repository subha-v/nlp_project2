import glob
import os
import io
import random
from utils import name_to_tensor
from utils import * #this imports everything
import torch


class NamesDataset:
    def __init__(self, dataset_path):
        self.country_to_names = {}
        filenames = glob.glob(dataset_path)
        for fn in filenames:
            basename = os.path.basename(fn)
            country = os.path.splitext(basename)[0] #gives the country name without the .txt at the end
            names = io.open(fn, encoding='utf-8').read().strip().split('\n') #this creates a list with all the names in it
            names = [name.lower() for name in names] #this makes everything lowercase
            self.country_to_names[country] = names #map country to names

        self.countries = list(self.country_to_names.keys()) #just gives the labels greek and korean in a list
        self.num_countries = len(self.countries) #get length of list of countries

    #each time we get a random name and country
    def get_random_sample(self):
        rand_country_idx = random.randint(0,self.num_countries-1) #gives a random country index
        country = self.countries[rand_country_idx]

        rand_name_idx = random.randint(0,len(self.country_to_names[country])-1)
        name = self.country_to_names[country][rand_name_idx] #u have to add self because its refering to the object you pass through

        name_tensor = name_to_tensor(name)
        country_tensor = torch.tensor([self.countries.index(country)], dtype=torch.long) #we want to make sure the datatype is long  

        return country, name, country_tensor, name_tensor

    def output_to_country(self, output):
        country_idx = torch.argmax(output).item() #which value is bigger?
        return self.countries[country_idx] #return the country