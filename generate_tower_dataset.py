import os
import json
import numpy as np

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

for i in range(COUNT):
    env = TowerExample(map_name="multi_step")
    state = env.reset(record=False)
    #print("Possible reward types:", env.reward_types())
    #print("Possible actions:", env.actions())
    #print("Action description", env.action_desc())
    #print('Reward: {}'.format(state.reward))

    # Save game state
    filename = 'towers/images/{:09d}.png'.format(i)
    imutil.show(state.state.transpose((2,0,1)), filename=filename)

    # Compute reward for each of 4 possible actions
    rewards = []
    act = env.new_action()
    for tower_id in range(1, 5):
        act.attack_quadrant(tower_id) # attack tower 2
        state = env.act(act)
        rewards.append(state.reward)
        state = env.reset()
    print('Got rewards: {}'.format(rewards))

    examples.append({
        'filename': filename,
        'label': int(np.argmax(rewards)),
    })

with open('towers.dataset', 'w') as fp:
    for example in examples:
        fp.write(json.dumps(example) + '\n')
