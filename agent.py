import torch
import random
import numpy as np
from collections import deque
from buffer import PrioritizedReplayBuffer
from game import SnakeGameAI, Direction, Point
from model import CNN_QNet, QTrainer
from plot import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
EPSILON_DECAY = 100000
MIN_EPSILON = 0.01

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 1 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = PrioritizedReplayBuffer(capacity=MAX_MEMORY, alpha=0.6)
        tam = 11
        self.model = CNN_QNet(tam, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def get_state(self, game):
        snake = torch.zeros((18, 16))
        # food = torch.zeros((18, 16))
        
        # food[int(game.food.x//20)-1, int(game.food.y//20)-1] = 1
        # head = game.snake[0]
        # left =  head.x//20-1
        # right = 18 - head.x//20 + 1
        # up = head.y//20-1
        # down = 16 - head.y//20 + 1
        # # snake[int(head.x//20)-1, int(head.y//20)-1] = 1
        # for point in game.snake[1:]:
        #     x = int(point.x//20)-1
        #     y = int(point.y//20)-1
        #     if(x == left):
        #         if y > up:
        #             up = min()
            
        #     snake[x,y] = 1
        # state = torch.stack([snake, food])
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN
        #state = torch.concat(torch.tensor([game.food.x, game.food.y,]), snake.flatten())
        state = torch.tensor([
            # # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y,  # food down
            
            
            ], dtype=torch.float)
        
        # #food = torch.tensor([game.food.x,game.food.y])
        # snake = snake.flatten()
        # state = torch.concat([aux, snake], 0)
        # print
        
        # state = torch.tensor(state, dtype=torch.float)
        
        return state

    def remember(self, state, action, reward, next_state, done):
        self.memory.push((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached


    def train_memory(self, frame_idx, batch_size=BATCH_SIZE):
        
        self.trainer.train_step(self.memory, frame_idx, batch_size)
        


    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
       
        final_move = [0,0,0]
        
        if (np.random.uniform() < self.epsilon):# and self.mode == 0:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            
            prediction = self.model(state.unsqueeze(0))
            move = prediction.argmax().item()
            final_move[move] = 1
            
        
        self.epsilon = max(MIN_EPSILON, self.epsilon * np.exp(-1.0 / EPSILON_DECAY))
            
        

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    count = 0
    while True:
        count += 1
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        

        # remember
        agent.remember(state_old, np.argmax(final_move), reward, state_new, done)

        
        if(count > 10000):
            agent.train_memory(count, 2)

        if(count % 1000 == 0):
            agent.trainer.update_target()
            
        if done:
            
            game.reset()
            agent.n_games += 1
            if(count > 10000):
                agent.train_memory(count)
            
            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()