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
from pathlib import Path
from copy import deepcopy

# Dash-related imports
import dash
from dash.dependencies import Input, Output, State
from dash import dash_table
from dash import dcc
import dash_daq as daq
from dash import html
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

# ============================
# 	    LOADING DATA
# ============================
# Below, we're going to load data into memory that's used in various methods

# Load the default parameters
default_params = {}
with open("param_defaults.json", "r") as param_json:
    default_params = json.load(param_json)

# Check to see if the "image_prompts" folder is in the file structure (and create it if it's not)
image_prompt_path = Path("image_prompts/")
if (not image_prompt_path.exists()):
    Path.mkdir(image_prompt_path)

# Check to see if the "config_files" folder is in the file structure (and create it if it's not)
config_file_path = Path("config_files/")
if (not config_file_path.exists()):
    Path.mkdir(config_file_path)

# Loading the Parameter Explanations Excel file
param_explanations_df = pd.read_excel("Parameter Explanations.xlsx")

# ============================
# 	    GENERAL METHODS
# ============================
# All of the code below are more general methods I can use throughout
# the rest of this app's code-base

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

                    # This is the "header" of the Text Prompt; it contains the title of the box, as well as a close button
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

                    # This is the Text Area for the Text Prompt
                    dbc.Row(
                        children=[
                            dbc.Col(
                                children=[
                                    dcc.Textarea(
                                        id={"type": "text_prompt_input", "index": prompt_num},
                                        style={"width": "100%", "resize": "none", "height": "125px"}
                                    )
                                ],
                                width=12
                            )
                        ],
                        style={"marginBottom": "10px"}
                    ),

                    # This is the Prompt Strength row
                    dbc.Row(
                        children=[
                            dbc.Col(
                                children=[
                                    html.Div(
                                        children=[
                                            html.B("Prompt Strength:", style={"display": "inline-block", "marginRight": "10px"}),
                                            dcc.Input(type="number",
                                                      value=1,
                                                      id={"type": "prompt_strength", "index": prompt_num},
                                                      style={"display": "inline-block", "width": "100px"})
                                        ],
                                    )
                                ],
                                width=12,
                            )
                        ],
                        style={"marginBottom": "15px"}
                    ),

                    # This is the Prompt Influence row
                    dbc.Row(
                        children=[
                            dbc.Col(
                                children=[
                                    html.Div(
                                        children=[
                                            html.B("Prompt Influence:",
                                                   style={"display": "inline-block", "marginRight": "10px"}),
                                            dbc.Progress(id={"type": "prompt_influence", "index": prompt_num})
                                        ],
                                    )
                                ],
                                width=12
                            )
                        ],
                        style={"marginBottom": "20px"}
                    )
                ],
                style={"border": "1px solid blue", "height": "300px", "padding": "10px"},
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

                    # This is the "header row", which contains a Prompt title and a close button
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

                    # This is the "Image Upload" box
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

                    # This is the Prompt Strength input
                    dbc.Row(
                        children=[
                            dbc.Col(
                                children=[
                                    html.Div(
                                        children=[
                                            html.B("Prompt Strength:",
                                                   style={"display": "inline-block", "marginRight": "10px"}),
                                            dcc.Input(type="number",
                                                      id={"type": "prompt_strength", "index": prompt_num},
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

                    # This is the Prompt Influence progress bar
                    dbc.Row(
                        children=[
                            dbc.Col(
                                children=[
                                    html.Div(
                                        children=[
                                            html.B("Prompt Influence:",
                                                   style={"display": "inline-block", "marginRight": "10px"}),
                                            dbc.Progress(id={"type": "prompt_influence", "index": prompt_num})
                                        ],
                                    )
                                ],
                                width=12
                            )
                        ],
                        style={"marginBottom": "10px"}
                    )
                ],
                style={"border": "1px solid red", "height": "300px", "padding": "10px"},
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
                                   "height": "300px"}
                        )
                    ],
                    width=4
                ),

    return component


# This method will add in the "computed" parameters when given the "chosen" parameters
def add_computed_parameters(chosen_params):

    # Extract a couple of important values from chosen_params
    steps = chosen_params["steps"]
    width = chosen_params["width"]
    height = chosen_params["height"]
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


# This method will generate a configuration file based on the settings that the user has selected
def generate_config(prompts, scores, settings):

    # First, we can step through each of the prompts in the prompts dict to sort them
    user_text_prompts = {}
    user_image_prompts = {}
    for promptKey, promptDict in prompts.items():
        prompt_type = promptDict["type"]
        prompt_score = scores[promptKey]
        if (prompt_type) == "text":
            user_text_prompts[promptKey] = {"value": promptDict["value"].strip(), "score": prompt_score}
        elif (prompt_type) == "image":
            user_image_prompts[promptKey] = {"value": promptDict["value"].strip(), "score": prompt_score}

    # First, we'll need to generate the text_prompt dictionary
    text_prompt_list = []
    for prompt_key, text_prompt in user_text_prompts.items():
        text_prompt_str = f"{text_prompt['value']}:{text_prompt['score']}"
        text_prompt_list.append(text_prompt_str)
    text_prompt_dict = {}
    if (len(text_prompt_list) > 0):
        text_prompt_dict = {"0": text_prompt_list}

    # Now, we'll need to generate the image_prompt dictionary
    image_prompt_list = []
    for prompt_key, image_prompt in user_image_prompts.items():
        image_prompt_str = f"{image_prompt_path.resolve()/Path(image_prompt['value'])}:{image_prompt['score']}"
        image_prompt_list.append(image_prompt_str)
    image_prompt_dict = {}
    if (len(image_prompt_list) > 0):
        image_prompt_dict["0"] = image_prompt_list

    # Now, we'll generate the settings based on what the user had specified
    final_settings = deepcopy(default_params)
    final_settings["text_prompts"] = text_prompt_dict
    final_settings["image_prompts"] = image_prompt_dict
    for setting_name, setting_value in settings.items():
        if (isinstance(setting_value, str)):
            setting_value = setting_value.strip()
        final_settings[setting_name] = setting_value

    # Add the "calculated options" based on the options in these final settings
    final_settings = add_computed_parameters(final_settings)

    # Now, save this configuration JSON in the config_files folder
    file_name = f"{str(config_file_path)}/{final_settings['batch_name']}.json"
    with open(file_name, "w") as config_json_file:
        json.dump(final_settings, config_json_file, indent=4)


# This method will parse the contents of an uploaded config file
def parse_uploaded_config(uploaded_config_data):

    content_type, content_string = uploaded_config_data.split(',')

    decoded = base64.b64decode(content_string)

    decoded_data = json.loads(decoded)

    return decoded_data


# This method will create a label, input, and Tooltip
def labeled_input_with_tooltip(parameter_name, parameter_label,
                               default_value, input_type,
                               tooltip_text, **input_kwargs):

    # Indicating whether the input_component is a Boolean or a Numerical / String input
    if (input_type == "boolean"):
        if (default_value):
            input_component = dbc.Checklist(id=f"{parameter_name}_input", value=[0], **input_kwargs)
        else:
            input_component = dbc.Checklist(id=f"{parameter_name}_input", value=[], **input_kwargs)
    else:
        input_component = dbc.Input(id=f"{parameter_name}_input", value=default_value,
                                    type=input_type, **input_kwargs)

    return html.Div(
        id=f"{parameter_name}_input_group",
        children=[
            dbc.Label(
                children=[html.B(parameter_label)],
                id=f"{parameter_name}_label"
            ),
            input_component,
            dbc.Tooltip(
                target=f"{parameter_name}_label",
                children=[
                    tooltip_text
                ]
            )
        ],
        style={}
    )


# This will create a parameter input using all of the different information from the Parameter Explanations.xlsx file
def parameter_input(param_name, **input_kwargs):

    # Query the parameter DataFrame to find this parameter
    param_df_row = param_explanations_df.query("parameter_name==@param_name").iloc[0]
    label = param_df_row["label"]
    param_type = param_df_row["type"]
    default = param_df_row["default_value"]
    description = param_df_row["description"]
    return labeled_input_with_tooltip(param_name, label, default, param_type, description, **input_kwargs)


# This method will load a dictionary of inputs from the Parameter Explanations.xslx file
def load_parameter_defaults():

    # Iterate through each of the rows and grab the default value
    defaults = {}
    for row in param_explanations_df.itertuples():
        defaults[row.parameter_name] = row.default_value

    # Return this dictionary
    return defaults


