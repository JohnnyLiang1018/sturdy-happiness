from ensurepip import bootstrap
import requests
from kafka import KafkaConsumer, KafkaProducer
import base64
import pickle
from torch import onnx
import torch
import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.init as init
import base64
import rlkit.torch.pytorch_util as ptu


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


class ServerRequest():

    def __init__(self):
        self.server_url = "http://localhost:8081"
        self.topic = ""
        self.producer = KafkaProducer(
            bootstrap_servers='127.0.0.1:9092',
            client_id='ML'
        )
        self.num_trajectory = 0

        self.consumer = KafkaConsumer(
            bootstrap_servers=':9092',
            group_id = "robot-test"
        )
    
    def training_request(self, agents, env, numEnsemble, max_path_length, iteration, ber_mean):
        models = []
        dummy_input = torch.randn(5, requires_grad=True)
        print("dim", agents[0].input_size)
        for agent in agents:
            agent.to(torch.device("cpu"))
            torch.onnx.export(agent, dummy_input, "agent.onnx")
            with open('agent.onnx','rb') as handle:
                encode_byte = base64.b64encode(handle.read())
            encode_string = encode_byte.decode("ascii")
            models.append(encode_string)
            agent.to(ptu.device)
        
        json = {"numEnsemble": numEnsemble, "policy": models, "iteration": iteration, "epLength": max_path_length}
        response = requests.post(self.server_url+'/request',json=json)
        self.topic = response.text
        print("topic", self.topic)
        self.consumer.subscribe(self.topic)

        # consumer = KafkaConsumer(
        #     self.topic,
        #     bootstrap_servers=':9092',
        #     group_id = "robot"
        # )

        paths = []
        count = 0
        print(f"Waiting for {iteration} samples")
        for message in self.consumer:
            if message.value is None:
                continue
        # while True:
        #     records = consumer.poll(timeout_ms=1000)
        #     for topic_data, consumer_records in records.items():
        #         for consumer_record in consumer_records:
        #             print("Received message: " + str(consumer_record.value.decode('utf-8')))
            reading = base64.b64decode(message.value)
            json = pickle.loads(reading)
            masks = []
            obs = json['states']
            for i in range(len(obs)):
                mask = torch.bernoulli(torch.Tensor([ber_mean]*numEnsemble)) ##
                if mask.sum() == 0:
                    rand_index = np.random.randint(numEnsemble, size=1)
                    mask[rand_index] = 1
                mask = mask.numpy()
                masks.append(mask)

            actions = json['actions'][:,:1]
            next_obs = json['next_obs']
            rewards = json['rewards']
            terminals = json['dones']
            env_info = json['env_info']
            agent_info = json['agent_info']

            if len(actions.shape) == 1:
                actions = np.expand_dims(actions, 1)
            observations = np.array(obs)
            if len(observations.shape) == 1:
                observations = np.expand_dims(observations, 1)
                next_obs = np.array([json['next_obs']])
            masks = np.array(masks)

            size = json['size']
            print("obs", json['states'])
            print("next obs", json['next_obs'])
            self.plot_trajectory(np.vstack((obs,np.expand_dims(next_obs,0))))
            next_obs = np.vstack(
                (
                    obs[1:, :],
                    np.expand_dims(next_obs, 0)
                )
            )
            # if (iteration - count) < size:
            #     remaining = iteration-count
            #     observations = observations[:remaining]
            #     actions = actions[:remaining]
            #     rewards = rewards[:remaining]
            #     terminals = terminals[:remaining]
            #     agent_info = agent_info[:remaining]
            #     env_info = env_info[:remaining]

            paths.append(dict(
                observations=observations,
                actions=actions,
                rewards=rewards.reshape(-1, 1),
                next_observations=next_obs,
                terminals=terminals.reshape(-1, 1),
                agent_infos=agent_info,
                env_infos=env_info,
                masks=masks,
            ))
            # print("state", states)
            # print("actions", actions)
            # print("received steps:" , size)

            count += int(size)
            print(f"Get {int(size)} samples, {iteration - count} samples remaining")
            if count >= iteration:
                break

        return paths

    
    def evaluate(self, agent, max_path_length, iteration):
        dummy_input = torch.randn(5, requires_grad=True)
        models = []
        agent.to(torch.device("cpu"))
        torch.onnx.export(agent, dummy_input, "agent.onnx", verbose=True)
        with open('agent.onnx','rb') as handle:
            encode_byte = base64.b64encode(handle.read())
        encode_string = encode_byte.decode("ascii")
        models.append(encode_string)
        json = {"numEnsemble": 1, "policy": models, "iteration": iteration, "epLength": max_path_length}
        response = requests.post(self.server_url+'/request',json=json)
        self.topic = response.text
        print("topic", self.topic)
        self.consumer.subscribe(self.topic)

        r_avg = 0
        count = 0
        for message in self.consumer:
            if message.value is None:
                continue

            reading = base64.b64decode(message.value)
            json = pickle.loads(reading)
            rewards = json['rewards']
    
            for r in rewards:
                r_avg += (r / iteration)
                count += 1
                if count == iteration:
                    return r_avg
            
            
    def training_data_upload(self, dict, epoch):
        # json = {"policyLoss": str(dict['Policy_loss']), "epoch": epoch, "criticLoss": str(dict['Critic_loss'])}
        # response = requests.post(self.server_url+'/train', json=json)
        # print("train progress sent")
        value = str(epoch) + ',' + str(dict['Policy_loss']) + ',' + str(dict['Critic_loss'])
        byte = value.encode('utf-8')
        # byte = base64.b64encode(pickle.dumps(value))
        self.producer.send(topic='trainData', value=byte)

    def plot_trajectory(self, obs):
        for i in range(len(obs)-1):
            x, y = obs[i][:2]
            x_, y_ = obs[i+1][:2]
            target_x, target_y = obs[i][3:]
            target_x_, target_y_ = obs[i+1][3:]
            if i == 0:
                plt.scatter(x,y,s=100, color="red", marker="o")
                plt.scatter(target_x,target_y,s=100, color="blue", marker="o")
    #         target_x, target_y = obs[i][3:]
            plt.arrow(x,y,x_-x,y_-y,head_width=2)
            plt.arrow(target_x,target_y, target_x_-target_x, target_y_-target_y,head_width=2)
        
        plt.xlim([-10,160])
        plt.ylim([-10,160])
        # plt.savefig(fname=f'trajectory_{self.num_trajectory}')
        self.num_trajectory += 1
        plt.show(block=True)