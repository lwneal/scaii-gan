import os
import json
import numpy as np
import random

from tqdm import tqdm
from gnomehat import imutil
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

os.makedirs('towers/images', exist_ok=True)

examples = []

for i in tqdm(range(COUNT)):
    env = TowerExample(map_name="multi_step")
    state = env.reset(record=False)

    # Save game state
    filename = 'towers/images/{:09d}.png'.format(i)
    imutil.show(state.state.transpose((2,0,1)), filename=filename)

    # Compute reward for one randomly-selected action
    tower_id = random.choice(range(1, 5))
    act = env.new_action()
    act.attack_quadrant(tower_id)
    state = env.act(act)
    print('Took action {}, got reward {}'.format(tower_id, state.reward))

    examples.append({
        'filename': filename,
        'action': tower_id,
        'value': state.reward,
    })

with open('towers.dataset', 'w') as fp:
    for example in examples:
        fp.write(json.dumps(example) + '\n')
