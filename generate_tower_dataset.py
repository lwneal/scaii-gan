import os
import json
import numpy as np
import random

from tqdm import tqdm
import imutil
from scaii.env.sky_rts.env.scenarios.tower_example import TowerExample

COUNT = 100 * 1000

state_channel_names = [
    "Health",
    "Agent Location",
    "Small Towers",
    "Big Towers",
    "Friend",
    "Enemy",
]

action_names = {
    1: 'bottom_right',
    2: 'bottom_left',
    3: 'top_right',
    4: 'top_left',
}


os.makedirs('towers/images', exist_ok=True)

examples = []

def state_to_pixels(state):
    pixels = state.state.transpose((2,0,1))
    # HP ranges from [0,.13], rescale it here
    pixels[0] *= 6
    # Rescale the full image to [0,255]
    return pixels * 255


def state_to_reward(state):
    return {k: v for k, v in state.typed_reward.items()}


for i in tqdm(range(COUNT)):
    #env = TowerExample(map_name="multi_step")
    env = TowerExample(map_name="tower_example")
    state = env.reset(record=False)

    pixels = state_to_pixels(state)

    filename = 'towers/images/{:09d}.png'.format(i)
    imutil.show(pixels, filename=filename, normalize_color=False)

    # Compute reward for one randomly-selected action
    tower_id = random.choice(range(1, 5))
    act = env.new_action()
    act.attack_quadrant(tower_id)
    next_state = env.act(act)
    print('Took action {}, got reward {}'.format(
        action_names[tower_id], state_to_reward(next_state)))

    new_pixels = state_to_pixels(next_state)

    next_filename = 'towers/images/{:09d}_{}.png'.format(i, tower_id)
    imutil.show(new_pixels, filename=next_filename, normalize_color=False)

    examples.append({
        'filename': filename,
        'action': action_names[tower_id],
        'next_filename': next_filename,
        'value': state.reward,
        'fold': 'train' if i % 10 else 'test'
    })

with open('towers.dataset', 'w') as fp:
    for example in examples:
        fp.write(json.dumps(example) + '\n')
