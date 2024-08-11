#import imageio
import os
import sys
import gymnasium as gym
import flappy_bird_env
import pygame
from PIL import Image


def convert(state):
    """
    state: returned tuple
    """
    image = Image.fromarray(new_state)
    image = image.resize((84, 84)).convert("L")
    # Show the processed image
    # image.show()
    return image

if __name__ == '__main__':
    #main()
    env = gym.make("FlappyBird-v0", render_mode="rgb_array_list")
    state, _ = env.reset()
    new_state,reward,terminated,truncated,info = env.step(0)
    image = convert(new_state)
    image.show()
    print(image.size)

