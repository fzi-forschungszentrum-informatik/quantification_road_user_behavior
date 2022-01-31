import os
import math
import numpy as np
import itertools
import math
import uuid
import time
import argparse
import csv
import sys
import matplotlib.pyplot as plt
import seaborn as sb
from scipy import stats
import time
import multiprocessing as mp

class CreateDistribution(object):
    
    def __init__(self, name):
        self.name = name

    def get_paths(self, path):
        x = 0
        path_list = []
        for root, dirs, files in os.walk(os.path.abspath(path)):
            for file in files:
                # if len(path_list) > 100:
                #     return path_list
                if file.endswith("Average_rule_score.csv"):
                    path_list.append(os.path.join(root, file))
        return path_list


    def read_directory(self, path):
        slack = []

        path = path + "/files"

        file_list = self.get_paths(path)
        print("Collected paths!")


        pool = mp.Pool(mp.cpu_count())

        results = []
        results = pool.starmap(self.read_state, [[file_list[x]] for x in range(len(file_list))])

        pool.close()

        print("Calculatetd individual states!")
        results = np.array(results)
        results = np.concatenate(results)
        print("Stacked results!")

        # for state in file_list:
        #     tmp = self.read_state(state)
        #     slack = slack + tmp

        return np.array(results).astype(np.float)
    

    def read_state(self, path):
        slack_list = []
        with open(path, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in spamreader:
                if not row[0] == "nan" and not row[0].startswith("R"):
                    slack_list.append(row[0])
            
            return slack_list


    def get_Distribution(self, slack):

        mu = np.sum(slack) / len(slack)

        square = []
        for x in slack:
            p = (x - mu)**2
            square.append(p)
        square = np.array(square).astype(np.float)
        
        var = np.sum(square) / len(square)
        sigma = var**0.5

        mu = int(mu * 1000)/ 1000.0
        sigma = int(sigma * 1000)/ 1000.0
        
        return mu, sigma


    def generate_hist(self, path):
        log = True

        slack_list = self.read_directory(path)

        mu, sigma = self.get_Distribution(slack_list)

        slack_no_one = [] # eliminate slacks == 1.0 to retrieve clean plot
        for slack in slack_list:
            if not slack == 1.:
                slack_no_one.append(slack)
        print("Eliminated ones!")

        slack_list = np.array(slack_list).astype("float32")
        slack_no_one = np.array(slack_no_one).astype("float32")

        n, bins, patches = plt.hist(x=slack_list, bins=20, log=log, color='#0504aa',
                                    alpha=0.7, rwidth=0.85)

        plt.title(self.name + " Rule")
        plt.ylabel("# number")
        plt.xlabel("Slack value")
        plt.grid(axis='y', alpha=0.75)
        plt.text(0.15, n.max()*0.75, s="$\mu=$" + str(mu) + "$,\sigma=$" + str(sigma))
        plt.savefig(path + "Distribution.svg", bbox_inches='tight', pad_inches = 0)
        plt.show(block=True)
        plt.close() #closes all output afterwards

        print("Showed first plot!")
        
        n, bins, patches = plt.hist(x=slack_no_one, bins=20, log=log, color='#0504aa',
                                    alpha=0.7, rwidth=0.85)

        plt.title(self.name + " Rule ex. score=1")
        plt.ylabel("# number")
        plt.xlabel("Slack value")
        plt.grid(axis='y', alpha=0.75)
        plt.text(0.15, n.max()*0.75, s="$\mu=$" + str(mu) + "$,\sigma=$" + str(sigma))
        plt.savefig(path + "Distribution_no_one.svg", bbox_inches='tight', pad_inches = 0)
        plt.show(block=True)
        plt.close()


        n, bins, patches = plt.hist(x=slack_no_one, density=True, bins=20, log=log, color='#0504aa',
                            alpha=0.7, rwidth=0.85)

        kde1 = stats.gaussian_kde(slack_no_one)
        xx = np.linspace(0, 1, 1000)
        plt.plot(xx, kde1(xx), color="orange", label="pdf")

        # data_space = np.linspace(slack_no_one.min(), slack_no_one.max(), 100)
        # evaluated = kde1.evaluate(data_space)
        # evaluated = evaluated + 0.25
        # plt.plot(data_space, evaluated, color="m", label="~pdf (+0.5)")

        # plt.ylim(0,2.4)
        plt.title(self.name + " Rule")
        plt.ylabel("Probability Density")
        plt.xlabel("Slack value")
        plt.legend()
        plt.grid(axis='y', alpha=0.75)
        plt.text(0.15, n.max()*0.6, s="$\mu=$" + str(mu) + "$,\sigma=$" + str(sigma))
        plt.savefig(path + "PDF_no_one.svg", bbox_inches='tight', pad_inches = 0)
        plt.show(block=True)
        plt.close()

        n, bins, patches = plt.hist(x=slack_list, density=True, bins=20, log=log, color='#0504aa',
                    alpha=0.7, rwidth=0.85)

        kde1 = stats.gaussian_kde(slack_list)
        xx = np.linspace(0, 1, 1000)
        plt.plot(xx, kde1(xx), color="orange", label="pdf")

        # data_space = np.linspace(slack_list.min(), slack_list.max(), 100)
        # evaluated = kde1.evaluate(data_space)
        # evaluated = evaluated + 0.25
        # plt.plot(data_space, evaluated, color="m", label="~pdf (+0.5)")

        # plt.ylim(0,2.4)
        plt.title(self.name + " Rule ex. score=1")
        plt.ylabel("Probability Density")
        plt.xlabel("Slack value")
        plt.legend()
        plt.grid(axis='y', alpha=0.75)
        plt.text(0.15, n.max()*0.6, s="$\mu=$" + str(mu) + "$,\sigma=$" + str(sigma))
        plt.savefig(path + "PDF.svg", bbox_inches='tight', pad_inches = 0)
        plt.show(block=True)
        plt.close()

        sb.histplot(data=slack_list, log_scale=log, bins=20, kde=True, stat='probability')
        plt.title(self.name + " Rule")
        plt.savefig(path + "sb_probability.svg", bbox_inches='tight', pad_inches = 0)
        plt.xlabel("Slack value")
        plt.show(block=True)
        plt.close()

        sb.histplot(data=slack_no_one, log_scale=log, kde=True, bins=20, stat='probability')
        plt.title(self.name + " Rule ex. score=1")
        plt.savefig(path + "sb_probability_no_one.svg", bbox_inches='tight', pad_inches = 0)
        plt.xlabel("Slack value")
        plt.show(block=True)
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="")
    parser.add_argument("--name", type=str, default="Distance")

    args = parser.parse_args()

    # cr = CreateDistribution()
    # print("mu, sigma: " + str(cr.get_Distribution(path=args.path)))

    cb = CreateDistribution(args.name)
    cb.generate_hist(args.path)