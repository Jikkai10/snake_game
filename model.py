import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

GAMMA = 0.99

class CNN_QNet(nn.Module):
    # def __init__(self, input_size, output_size):
    #     super().__init__()
    #     self.cnn = nn.Sequential(
    #         nn.Conv2d(2, 16, kernel_size=3, padding=1),  
    #         nn.ReLU(),
            
    #         nn.Conv2d(16, 32, kernel_size=3, padding=1), 
    #         nn.ReLU(),
    #         #nn.MaxPool2d(2,2),
    #         nn.Conv2d(32, 32, kernel_size=3, padding=1), 
    #         nn.ReLU(),
            
    #         nn.Flatten()
    #     )
        
    #     dummy_input = torch.zeros(1, 2, 18, 16)
    #     flat = self.cnn(dummy_input).shape[1]
        
    #     self.linear = nn.Sequential(
    #         nn.Linear(flat, 256),
    #         nn.ReLU(),
    #         nn.Linear(256, output_size)
    #     )
        

    # def forward(self, x):
    #     x = self.cnn(x)
    #     x = self.linear(x)
    #     #print(x.shape)
    #     return x
    
    def __init__(self, input_size, output_size, hidden_size=256):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size).cuda()
        self.linear2 = nn.Linear(hidden_size, output_size).cuda()

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.target = model
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        
    def train_step(self, buffer, frame_idx, batch_size, gamma=GAMMA, beta_start=0.4, beta_frames=1000000):
        if len(buffer) < batch_size:
            return

        beta = min(1.0, beta_start + frame_idx * (1.0 - beta_start) / beta_frames)
        transitions, idxs, weights = buffer.sample(batch_size, beta)

        states, actions, rewards, next_states, dones = zip(*transitions)

        states = torch.stack(states)
        
        #extras = torch.stack(extras)
        actions = torch.tensor(actions).unsqueeze(1)
        rewards = torch.tensor(rewards).unsqueeze(1)
        next_states = torch.stack(next_states)
        #new_extras = torch.stack(new_extras)
        dones = torch.tensor(dones).unsqueeze(1)
        weights = torch.tensor(weights, dtype=torch.float32).unsqueeze(1)

        q_values = self.model(states).gather(1, actions)
        with torch.no_grad():
            next_actions = self.model(next_states).argmax(1, keepdim=True)
            next_q = self.target(next_states).gather(1, next_actions)
            target_q = rewards + gamma * next_q * (~dones)

        td_errors = target_q - q_values
        loss = (weights * td_errors.pow(2)).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    

        buffer.update_priorities(idxs, td_errors.detach().squeeze().cpu().numpy())
        
    def update_target(self):
        self.target.load_state_dict(self.model.state_dict())