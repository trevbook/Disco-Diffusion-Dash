# This file contains various helper methods for the Dash app

# Various import statements
import math
import json
import random
import pandas as pd
import numpy as np
import argparse
from PIL import Image
import numpy as np
import base64
import io



# Dash-related imports
import dash
from dash.dependencies import Input, Output, State
from dash import dash_table
from dash import dcc
import dash_daq as daq
from dash import html
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate


# This method will find a given ID in the row of components, and then return which index that
# component is in the list of children
def locate_id_index(id, row_children):

    # Iterate through all of the children, and then return the index of the child whose ID matches "id"
    for childIdx, child in enumerate(row_children):
        child_id = child["props"].get("id")
        if (child_id == id):
            return childIdx

    # If we haven't found the ID, then return None
    return None


# This method will generate a text prompt component
def generate_text_prompt_component(prompt_num):

    # Generate the component
    component = dbc.Col(
        id=f"prompt_col_{prompt_num}",
        children=[
            html.Div(
                children=[
                    dbc.Row(
                        children=[
                            dbc.Col(
                                children=[
                                    html.B("Text Prompt")
                                ],
                                width=10
                            ),
                            dbc.Col(
                                children=[
                                    dbc.Button(
                                        id={"type": "close_button", "index": prompt_num},
                                        children=["x"],
                                        size="sm",
                                        color="danger"
                                    ),
                                ],
                                width=2
                            )
                        ],
                        style={"marginBottom": "10px"}
                    ),
                    dbc.Row(
                        children=[
                            dbc.Col(
                                children=[
                                    dcc.Textarea(
                                        style={"width": "100%", "resize": "none", "height": "125px"}
                                    )
                                ],
                                width=12
                            )
                        ],
                        style={"marginBottom": "10px"}
                    ),
                    dbc.Row(
                        children=[
                            dbc.Col(
                                children=[
                                    html.Div(
                                        children=[
                                            html.B("Prompt Strength:", style={"display": "inline-block", "marginRight": "10px"}),
                                            dcc.Input(type="number",
                                                      value=1,
                                                      style={"display": "inline-block", "width": "100px"})
                                        ],
                                    )
                                ],
                                width=12,
                            )
                        ],
                        style={"marginBottom": "15px"}
                    ),
                    dbc.Row(
                        children=[
                            dbc.Col(
                                children=[
                                    html.Div(
                                        children=[
                                            html.B("Prompt Influence:",
                                                   style={"display": "inline-block", "marginRight": "10px"}),
                                            dbc.Progress(style={"display": "inline-block", "width": "100px"})
                                        ],
                                    )
                                ],
                                width=12
                            )
                        ],
                        style={"marginBottom": "20px"}
                    )
                ],
                style={"border": "1px solid blue", "height": "280px", "padding": "10px"},
            )
        ],
        width=4
    )

    # Return the component
    return component



# This method will generate an image prompt component
def generate_image_prompt_component(prompt_num):
    # Generate the component
    component = dbc.Col(
        id=f"prompt_col_{prompt_num}",
        children=[
            html.Div(
                children=[
                    dbc.Row(
                        children=[
                            dbc.Col(
                                children=[
                                    html.B("Image Prompt")
                                ],
                                width=10
                            ),
                            dbc.Col(
                                children=[
                                    dbc.Button(
                                        id={"type": "close_button", "index": prompt_num},
                                        children=["x"],
                                        size="sm",
                                        color="danger"
                                    ),
                                ],
                                width=2
                            )
                        ],
                        style={"marginBottom": "10px"}
                    ),
                    dbc.Row(
                        children=[
                            dbc.Col(
                                children=[
                                    dcc.Upload(
                                        id={"type": "image_prompt_upload", "index": prompt_num},
                                        children=["Upload Image"],
                                        style={"width": "100%", "height": "125px", "borderStyle": "dashed",
                                               "borderWidth": "1px", "textAlign": "center", "lineHeight": "125px",
                                               "cursor": "pointer"}
                                    )
                                ],
                                width=12
                            )
                        ],
                        style={"marginBottom": "10px"}
                    ),
                    dbc.Row(
                        children=[
                            dbc.Col(
                                children=[
                                    html.Div(
                                        children=[
                                            html.B("Prompt Strength:",
                                                   style={"display": "inline-block", "marginRight": "10px"}),
                                            dcc.Input(type="number",
                                                      value=1,
                                                      style={"display": "inline-block", "width": "100px"})
                                        ],
                                    )
                                ],
                                width=12
                            )
                        ],
                        style={"marginBottom": "15px",
                               "paddingTop": "6px"}
                    ),
                    dbc.Row(
                        children=[
                            dbc.Col(
                                children=[
                                    html.Div(
                                        children=[
                                            html.B("Prompt Influence:",
                                                   style={"display": "inline-block", "marginRight": "10px"}),
                                            dbc.Progress(style={"display": "inline-block", "width": "100px"})
                                        ],
                                    )
                                ],
                                width=12
                            )
                        ],
                        style={"marginBottom": "10px"}
                    )
                ],
                style={"border": "1px solid red", "height": "280px", "padding": "10px"},
            )
        ],
        width=4
    )

    # Return the component
    return component


# This method will convert a Data URL to a PIL Image
def data_url_to_image(base64_str):
    val = base64_str[22:]
    return Image.open(io.BytesIO(base64.decodebytes(bytes(val, "utf-8"))))



# This method will generate the "Add Prompt" buttons
def generate_add_prompt_buttons():

    component = dbc.Col(
                    children=[
                        html.Div(
                            children=[
                                dbc.ButtonGroup(
                                    children=[
                                        dbc.Button(
                                            id="add_text_prompt_button",
                                            children=["Add Text Prompt"],
                                            style={"width": "125px", "marginBottom": "10px"},
                                            color="primary"
                                        ),
                                        dbc.Button(
                                            id="add_image_prompt_button",
                                            children=["Add Image Prompt"],
                                            style={"width": "125px"},
                                            color="primary"
                                        )
                                    ],
                                    vertical=True,
                                    style={"marginTop": "70px"}
                                )
                            ],
                            style={"textAlign": "center",
                                   "border": "1px dotted black",
                                   "height": "280px"}
                        )
                    ],
                    width=4
                ),

    return component


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
        else:
            steps_per_checkpoint = steps + 10

    # Return the steps_per_checkpoint
    return steps_per_checkpoint


# This method will shift all of the prompts back one space
def shift_prompts_back_one(current_rows, affected_row, affected_idx):

    # Figure out which rows are unaffected
    unaffected_rows = current_rows[:(affected_row)]

    # Now, grab all of the Columns after the affected
    all_affected_cols = []
    affected_rows = current_rows[affected_row:]
    for row_idx, row in enumerate(affected_rows):
        cur_cols = row['props']['children']
        actual_row_idx = row_idx+affected_row
        for col_idx, col in enumerate(cur_cols):
            if (actual_row_idx == affected_row and col_idx == affected_idx):
                continue
            all_affected_cols.append(col)

    # Now, we need to create new rows using the columns that we've grabbed in
    new_rows = []
    new_row_amt = int(len(all_affected_cols) / 3) + 1
    for new_row in range(new_row_amt):
        start_idx = (new_row*3)
        end_idx = ((new_row+1)*3)
        if (start_idx >= len(all_affected_cols)):
            break
        col_subset = all_affected_cols[start_idx:end_idx]
        new_rows.append(dbc.Row(children=col_subset, style={"marginBottom": "25px"}))

    return_rows = unaffected_rows + new_rows

    return return_rows
