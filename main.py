import pickle
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import pickle
import copy
import random
import numpy as np


class Genome():
    def __init__(self):
        self.w1 = np.random.randn(3,6)
        self.b1 = np.random.randn(6,)
        self.w2 = np.random.randn(6,3)
        self.b2 = np.random.randn(3,)
        self.fitness = 0

    def out(self, distance, ypos, speed):
        inp = np.array([distance, ypos, speed], dtype=float)
        res = np.dot(inp,self.w1)
        res = res + self.b1
        res = np.tanh(res)
        res = np.dot(res,self.w2)
        res = res + self.b2
        res = self.softmax(res)
        return res

    def softmax(self,x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)


class Generation:
    def __init__(self):
        try:
            file = open("savedweights.pkl",'rb')
            self.genomes = pickle.load(file)
            print("Loading from the previous state")
        except:
            print("Creating new genomes")
            self.genomes = [Genome() for i in range(12)]

        self.fittest = []
        self.loosers = []

        self.browser = webdriver.Chrome()
        self.browser.get("chrome://dino")
        self.canvas = self.browser.find_element_by_id('t')
        self.canvas.send_keys(Keys.SPACE)

    def survive(self):
        for i,genome in enumerate(self.genomes):
            while(1):
                try:
                    ypos = (self.browser.execute_script('return Runner.instance_.horizon.obstacles[0].yPos'))
                    dist = (self.browser.execute_script('return Runner.instance_.horizon.obstacles[0].xPos')) - 24
                    speed = (self.browser.execute_script('return Runner.instance_.currentSpeed'))
                except:
                    dist=0
                    speed=0
                    ypos =0

                output = genome.out(dist/1000,ypos/100,speed/10)   #normalizing input
                #print(output)

                if(np.argmax(output) == 1):
                    self.canvas.send_keys(Keys.ARROW_UP)              #jump
                else:
                    if(np.argmax(output) == 2):
                        self.canvas.send_keys(Keys.ARROW_DOWN)          

                if self.browser.execute_script('return Runner.instance_.crashed') == True:
                    genome.fitness = int(self.browser.execute_script('return Runner.instance_.distanceRan'))
                    self.browser.execute_script('Runner.instance_.restart()')
                    break

        self.genomes.sort(key=lambda x: x.fitness, reverse=True)
        self.fittest = self.genomes[:4]
        self.loosers = self.genomes[4:]
        print(" Fitness = ", self.fittest[0].fitness)
        self.genomes = self.fittest


    def breed_and_mutate(self):
        while len(self.genomes) < 8:                    #breeding among the fittest
            g1 = np.random.choice(self.fittest)
            g2 = np.random.choice(self.fittest)
            self.genomes.append(self.cross_over(self.mutate(g1), self.mutate(g2)))
        while len(self.genomes) < 12:                   #breeding the fittest with the loosers
            g1 = np.random.choice(self.fittest)
            g2 = np.random.choice(self.loosers)
            self.genomes.append(self.cross_over(self.mutate(g1),self.mutate(g2)))

    def cross_over(self, g1, g2):
        new_genome = copy.deepcopy(g1)
        other_genome = copy.deepcopy(g2)
        for i in range(0,3):
            cut_loc = random.randint(0,5)       #cutting at random position
            for j in range(cut_loc):
                new_genome.w1[i][j],g2.w1[i][j] = g2.w1[i][j], new_genome.w1[i][j]

        for i in range(0,6):
            cut_loc = random.randint(0,2)        #cutting at random position
            for j in range(cut_loc):
                new_genome.w2[i][j],g2.w2[i][j] = g2.w2[i][j], new_genome.w2[i][j]
        return new_genome

    def mutate_weights(self, weights):
        if np.random.uniform(0, 1) < 0.2:
            print("An individual is mutated")
            return weights * (np.random.uniform(0, 1) - 0.5) * 3 + (np.random.uniform(0, 1) - 0.5)
        else:
            return 0

    def mutate(self,g):
        new_genome = copy.deepcopy(g)
        new_genome.w1 += self.mutate_weights(new_genome.w1)
        new_genome.w2 += self.mutate_weights(new_genome.w2)
        return new_genome



def main():
    gen = Generation()
    i=0
    while(1):
        print("GENERATION -> ",i+1,end='')
        gen.survive()
        gen.breed_and_mutate()
        i+=1


if __name__ == '__main__':
    main()