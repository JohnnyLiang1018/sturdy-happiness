import socket
from ast import literal_eval
import pickle
import requests

class Client(object):
    
    def __init__(self):
        # python
        self.host = socket.gethostname() 
        self.port = 4777 

        self.client_socket = socket.socket()
        self.client_socket.connect(('10.250.19.92', self.port))
        self.state_init = pickle.loads(self.client_socket.recv(1024))
        print(len(self.state_init))
        print(self.state_init[0])
        print(self.state_init[1])

        

    def request(self,actions):
        self.client_socket.send(pickle.dumps(actions))
        data = ''
        while data == '':
            data = pickle.loads(self.client_socket.recv(1024))
        # print("received " + str(data))
        obs = data[0]
        reward = data[1]
        done = data[2]
        info = data[3]

        return obs,reward,done,info
    

# client = Client()
# client.request(0)

# if __name__ == '__main__':
#     client_program()
