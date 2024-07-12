from collections import deque
import random

class ReplayMemory():
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)
    
    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)
    
    def HER(self, sample_size):
        experiences = self.sample(sample_size)
        list_experiences = []
        
        for state, action, next_state, reward, done, goal in experiences:
            goal = next_state
            
            if next_state == goal:
                HER_reward = 1.0
            else:
                HER_reward = 0.0
            
            list_experiences.append((state, action, next_state, HER_reward, done, goal)) # Yeni deneyimi listeye ekliyoruz.
        
        return experiences + list_experiences
    
    def __len__(self):
        return len(self.memory)