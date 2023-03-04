from multiprocessing import dummy
import requests
from kafka import KafkaConsumer
import base64
import pickle
from torch import onnx
import torch
import numpy as np


import torch.nn as nn
import torch.nn.init as init
import ast


class SuperResolutionNet(nn.Module):
    def __init__(self, upscale_factor, inplace=False):
        super(SuperResolutionNet, self).__init__()

        self.relu = nn.ReLU(inplace=inplace)
        self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        return x

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv4.weight)


class CollectionRequest():

    def __init__(self):
        self.server_url = "http://localhost:8081/request"
        self.topic = ""

    
    def request(self, agents, env, numEnsemble, max_path_length, iteration):
        models = []
        # dummy_input = torch.randn(1, 1, 224, 224, requires_grad=True)
        # torch_model = SuperResolutionNet(upscale_factor=3)
        # torch_model.eval()
        # dummy_input = torch.Tensor(env.reset())
        dummy_input = torch.randn(5, requires_grad=True)
        # torch.onnx.export(torch_model, dummy_input, "alexnet.onnx", verbose=True)

        for agent in agents:
            torch.onnx.export(agent, dummy_input, "agent.onnx", verbose=True)
            with open('agent.onnx','rb') as handle:
                encode_byte = base64.b64encode(handle.read())
            encode_string = encode_byte.decode("ascii")
            models.append(encode_string)
        json = {"numEnsemble": numEnsemble, "policy": models, "iteration": iteration, "ep_length": max_path_length}
        response = requests.post(self.server_url,json=json)
        self.topic = response.text
        print(response)
        print(response.text)

        consumer = KafkaConsumer(
            self.topic,
            bootstrap_servers=':9092',
            group_id = "robot"
        )

        states = []
        actions = []
        count = 0
        for message in consumer:
            ## index 0, id
            ## index 1, state list
            ## index 2, action list
            ## index 3, iteration
            reading = ast.literal_eval(str(message.value, 'utf-8'))
            print(reading)
            states.extend(reading[1])
            actions.extend(reading[2])
            print(states)
            print(actions)
            count += int(reading[3])
            if count > iteration:
                break

        return states, actions