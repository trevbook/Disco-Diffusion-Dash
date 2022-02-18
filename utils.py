# This file contains various helper methods for the Dash app

# Various import statements
import math
import json
import random
import pandas as pd
import numpy as np
import argparse


# This method will add in the "computed" parameters when given the "chosen" parameters
def add_computed_parameters(chosen_params):

    # Extract a couple of important values from chosen_params
    steps = chosen_params["steps"]
    width = chosen_params["width_height"][0]
    height = chosen_params["width_height"][1]
    set_seed = chosen_params["set_seed"]
    frames_skip_steps = chosen_params["frames_skip_steps"]
    max_frames = chosen_params["max_frames"]
    text_prompts = chosen_params["text_prompts"]
    image_prompts = chosen_params["text_prompts"]

    # This dict will hold the new params
    final_params = chosen_params

    # Calculate the side_x and side_y parameters from the [width, height] array
    final_params["side_x"] = (width//64)*64
    final_params["side_y"] = (height//64)*64

    # Grab the "seed" parameter from the set_seed
    final_params["seed"] = set_seed
    if (final_params["seed"] == "random_seed"):
        final_params["seed"] = random.randint(1, 10000)

    # Set the batchNum parameter to 0 (this will have to be changed in future runs)
    final_params["batchNum"] = 0

    # Calculate the diffusion_steps parameter
    final_params["diffusion_steps"] = (1000//steps)*steps if steps < 1000 else steps

    # Set the start_frame parameter to 0
    final_params["start_frame"] = 0

    # Change the intermediate_saves array
    final_params["intermediate_saves"] = [x for x in range(steps)]

    # Calculate the "steps_per_checkpoint" amount
    final_params["steps_per_checkpoint"] = calc_steps_per_checkpoint(final_params)

    # Calculate the skip_step_ratio
    final_params["skip_step_ratio"] = int(frames_skip_steps.rstrip("%")) / 100

    # Calculate the timestep_respacing
    final_params["timestep_respacing"] = f'ddim{steps}'

    # Calculate the calc_frames_skip_steps
    final_params["calc_frames_skip_steps"] = math.floor(steps * final_params["skip_step_ratio"])

    # Calculate n_batches
    n_batches = final_params["n_batches"]
    animation_mode = final_params["animation_mode"]
    final_params["n_batches"] = n_batches if animation_mode == 'None' else 1

    # Calculate max_frames
    max_frames = final_params['max_frames']
    final_params['max_frames'] = max_frames if animation_mode != "None" else 1

    # Change n_batches to an integer
    final_params["n_batches"] = int(final_params["n_batches"])



    # Return the "final_params" dictionary
    return final_params


# This is a helper function to calculate the steps_per_checkpoint amount
def calc_steps_per_checkpoint(chosen_params):

    # Extract certain values from the chosen_params dictionary
    intermediate_saves = chosen_params["intermediate_saves"]
    steps = chosen_params["steps"]
    skip_steps = chosen_params["skip_steps"]

    # Calculate the steps_per_checkpoint
    steps_per_checkpoint = None
    if type(intermediate_saves) is not list:
        if intermediate_saves:
            steps_per_checkpoint = math.floor((steps - skip_steps - 1) // (intermediate_saves + 1))
            steps_per_checkpoint = steps_per_checkpoint if steps_per_checkpoint > 0 else 1
            print(f'Will save every {steps_per_checkpoint} steps')
        else:
            steps_per_checkpoint = steps + 10

    # Return the steps_per_checkpoint
    return steps_per_checkpoint


