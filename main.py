import os
import sys
import pygame as pg
import pytmx
import numpy as np
import time

TILE_SIZE=16
NB_TILE_X=15
NB_TILE_Y=15
WIDTH=NB_TILE_X*TILE_SIZE
HEIGHT=NB_TILE_Y*TILE_SIZE

MAIN_DIR = os.path.split(os.path.abspath(__file__))[0]
DATA_DIR = os.path.join(MAIN_DIR, "data")

class SubjectOfTraining:
    def __init__(self):
        head=(0,0)
        self.segments=[head]

    def move(self,action):
        if action==0:#up
            self.direction = pg.math.Vector2(0,-1)
        if action==1:#down
            self.direction = pg.math.Vector2(0,1)
        if action==2:#right
            self.direction = pg.math.Vector2(1,0)
        if action==3:#left
            self.direction = pg.math.Vector2(-1,0)

        new_x_position = int(self.segments[0][0] + self.direction.x)
        new_y_position = int(self.segments[0][1] + self.direction.y)
        new_segment=(new_x_position,new_y_position)
        self.segments.insert(0,new_segment)
        self.segments.pop()
        return new_segment

    def reset(self):
        head=(0,0)
        self.segments=[head]


    def grow(self):
        tail = self.segments[-1]
        new_tail = (tail[0], tail[1])
        self.segments.append(new_tail)

class EnvironnementOfTrainning:
    def __init__(self):
        self.size_x=15
        self.size_y=15
        

class MasterOfTraining:
    def __init__(self):
        self.alpha=0.5
        self.gamma=0.95
        self.epsilon=0.1
        self.goal_of_subject=(5,5)
        
    def choose_action(self,state_of_subject,q_table):

        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice([0, 1, 2, 3])  # exploration
        else:
            return np.argmax(q_table[state_of_subject])  # exploitation 

    def update_q_table(self,q_table, previous_state, action, reward, new_state):

        q_table[previous_state][action] = (1 - self.alpha) * q_table[previous_state][action] + self.alpha * (reward + self.gamma * np.max(q_table[new_state]))
        
    def calculate_rewards(self,state_of_subject,size_map):
        reward=0
        done=False
        if state_of_subject == self.goal_of_subject:
            reward+=10
            done=True
        else:
            reward-=0.05

        if state_of_subject[0] >= size_map[0] or  state_of_subject[1] >= size_map[1] or state_of_subject[0] < 0 or state_of_subject[1] < 0:
            reward-=10
            done=True

        return reward,done

class RunProgramTraining:
    def __init__(self,episodes=10):
        self.master=MasterOfTraining()
        self.subject=SubjectOfTraining()
        self.env=EnvironnementOfTrainning()
        self.episodes=episodes
        self.q_table = np.zeros((self.env.size_x,self.env.size_y, 4))
        self.rewards=[]
        self.states = []  
        self.starts = []  
        self.steps_per_episode = [] 

    def train(self):
        steps = 0
        episode = 0
        size_map=(self.env.size_x,self.env.size_y)
        while episode < self.episodes:
            self.subject.reset()
            states_episode=[]
            total_reward = 0
            done=False
            while not done :
                previous_state=self.subject.segments[0]
                action = self.master.choose_action(previous_state,self.q_table)
                new_state = self.subject.move(action)
                states_episode.append(new_state)
                
                reward, done=self.master.calculate_rewards(new_state,size_map)
                total_reward += reward
                
                self.master.update_q_table(self.q_table, previous_state, action, reward, new_state)
                
                steps += 1 
                if done:
                    self.starts.append(len(states_episode))
                    self.rewards.append(total_reward)
                    self.steps_per_episode.append(steps)
                    self.states.append(states_episode)
                    steps = 0
                    episode += 1
        return self.rewards,self.states,self.starts,self.steps_per_episode

class PrintOneEpisode:
    def __init__(self,states):
        pg.init()
        self.screen = pg.display.set_mode((WIDTH,HEIGHT))
        pg.display.set_caption('Training')

        sprite_subject = os.path.join(DATA_DIR, "sprites", "snake.png")
        self.image_subject = pg.image.load(sprite_subject).convert_alpha()

        sprite_apple = os.path.join(DATA_DIR, "sprites", "apple.png")
        self.image_apple = pg.image.load(sprite_apple).convert_alpha()

        sprite_map = os.path.join(DATA_DIR, "map", "map_snake.tmx")
        self.map_data=pytmx.load_pygame(sprite_map)

        self.states=states


    def display_map(self):
        for layer in self.map_data.visible_layers:
            for x,y,gid in layer:
                tile=self.map_data.get_tile_image_by_gid(gid)
                if tile:
                    self.screen.blit(tile,(x*TILE_SIZE,y*TILE_SIZE))

    def run_animation(self,episode):
        if episode < 0 or episode >= len(self.states):
            print("Episode index out of range.")
            return
        for position in self.states[episode]:
            self.display_map()
            rect_subject=pg.Rect(position[0]*TILE_SIZE,position[1]*TILE_SIZE,TILE_SIZE,TILE_SIZE)
            rect_apple=pg.Rect(5*TILE_SIZE,5*TILE_SIZE,TILE_SIZE,TILE_SIZE)
            self.screen.blit(self.image_subject, rect_subject)
            self.screen.blit(self.image_apple,rect_apple)
            pg.display.flip()
            time.sleep(1)

if __name__ == '__main__':
    program = RunProgramTraining(episodes=10000)
    rewards,states,starts,steps_per_episode=program.train()
    #print(rewards)
    print(states[999])
    print_program=PrintOneEpisode(states)
    print_program.run_animation(999)

