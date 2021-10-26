import os
import torch
import random
import numpy as np


class Agent:
    def __init__(self):
        self.model = torch.load(__file__[:-8] + "/agent.pkl")
        self.model.eval()

    def act(self, state):
        with torch.no_grad():
            return self.model(torch.tensor(state)).argmax(-1).numpy()

