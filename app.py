# This Dash app was written by Trevor Hubbard; it's purpose is to provide a user-interface for the
# Disco Diffusion notebook (which is for CLIP-Guided Diffusion)

# ============================
# 	   IMPORT STATEMENTS
# ============================
# Below, you'll find a number of different import statements that
# import libraries necessary for the application to run properly

# General imports
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from copy import deepcopy
from PIL.ImageOps import contain
from pathlib import Path
from multiprocessing import Process, Queue


# Dash-related imports
import dash
from dash.dependencies import Input, Output, State, MATCH, ALL, ALLSMALLER
from dash import dash_table
from dash import dcc
import dash_daq as daq
from dash import html
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

# Imports related to additional modules I've written
import utils
from utils import generate_text_prompt_component, generate_image_prompt_component, generate_add_prompt_buttons, labeled_input_with_tooltip, parameter_input
from components import generate_config_window

# ============================
# 	    DASH APP SETUP
# ============================
# The code below helps load all of the necessary data for the app,
# and ensures that the Dash app object is set up properly

# Setting up the Dash app
app = dash.Dash(__name__,
                external_stylesheets=[dbc.themes.BOOTSTRAP])

# Declaring a title for the app
app.title = "Disco Diffusion"

# ============================
# 	    DATA STRUCTURES
# ============================
# The code below declares a couple of data structures that can be used throughout the app

# This 'counter' will enable the creation of prompts with monotonically increasing indices
prompt_idx = 0
config_idx = 0

# Check to see if the "image_prompts" folder is in the file structure (and create it if it's not)
image_prompt_path = Path("image_prompts/")
if (not image_prompt_path.exists()):
    Path.mkdir(image_prompt_path)

# Create the "component_to_settings" dictionary
component_to_settings = {
    "batch_name_input": "batch_name",
    "width_input": "width",
    "height_input": "height",
    "steps_input": "steps",
    "n_batches_input": "n_batches"
}

# This dict will store the current config queue
config_queue = {}
config_queue_status = {}

# This will open the file with parameter defaults
param_defaults = {}
with open("param_defaults.json", "r") as param_defaults_json:
    param_defaults = json.load(param_defaults_json)

# This will open the Parameter Explanations.xlsx file
param_explanations_df = pd.read_excel("Parameter Explanations.xlsx")
param_settings_inputs = [Input(f"{x.parameter_name}_input", "value") for x in param_explanations_df.itertuples()]

# ============================
# 	    GENERAL METHODS
# ============================
# All of the code below are more general methods I can use throughout
# the rest of this app's code-base


# ============================
# 	    DASH APP LAYOUT
# ============================
# Below, you'll find code that's directly defining the app's layout

# Setting up the main layout of the Dash app
app.layout = html.Div(

    children=[

        # These are some stores
        dcc.Store(id="prompt_store", data={}),
        dcc.Store(id="prompt_score_store", data={}),
        dcc.Store(id="settings_store", data=utils.load_parameter_defaults()),
        dcc.Store(id="config_queue_store", data={}),
        dcc.Store("misc_store_config_generation"),
        dcc.Store("misc_store_2"),
        dcc.Store(id="init_image_store", data=""),

        # This is an Interval, which will help with running actual Disco jobs
        dcc.Interval(id="queue_management_interval", interval=1000),

        # This is the container for the entire Dash app
        dbc.Container(

            children=[

                # This is the "header" row - it contains some introductory information about the app itself
                dbc.Row(
                    children=[
                        dbc.Col(
                            children=[
                                dcc.Markdown(
                                    children=[
                                        """
                              # Disco Diffusion Dash
                              This is a local version of [Disco Diffusion v4.1](https://colab.research.google.com/drive/1sHfRn5Y0YKYKi1k-ifUSBFRNJ8_1sa39?usp=sharing#scrollTo=LHLiO56OfwgD).
                              This UI was written by Trevor Hubbard using [Plotly Dash](https://plotly.com/dash/). 
                              Parameter explanations were taken from [Zippy's Disco Diffusion Cheatsheet](https://docs.google.com/document/d/1l8s7uS2dGqjztYSjPpzlmXLjl5PM3IGkRWI3IiCuK7g/edit?usp=sharing).
                              """
                                    ]
                                ),
                            ],
                            width=8,
                            style={"padding": "0px"}
                        ),
                        dbc.Col(
                            children=[
                                daq.BooleanSwitch(
                                    id="dev_mode_switch",
                                    on=False,
                                    label="Developer Mode",
                                    labelPosition="top"
                                ),
                                dbc.Tooltip(
                                    children=[
                                        "This will turn on 'Developer Mode', which exposes a number of advanced parameters to tweak."
                                    ],
                                    target="dev_mode_switch"
                                )
                            ],
                            width=4,
                            style={"padding": "0px"}
                        )
                    ],
                    style={"marginBottom": "20px"},
                    justify="between"
                ),

                # This is the "Configuration Settings Header"
                dbc.Row(
                    children=[
                        dbc.Col(
                            children=[
                                dbc.Row(
                                    children=[
                                        dcc.Markdown(
                                            children=[
                                                """
                                                ### Generate Configuration File
                                                Below, there are a number of different settings that you ought to specify; these 
                                                settings will be compiled together into a "configuration file", which can be 
                                                used to run Disco Diffusion. 
                                                """
                                            ]
                                        )
                                    ]
                                )
                            ],
                            width=12
                        )
                    ],
                    style={"marginBottom": "10px"}
                ),

                # This is the "Basic Settings" row - it contains various basic settings for the app
                dbc.Row(
                    children=[
                        html.Details([
                            html.Summary(
                                "Basic Settings"
                            ),
                            html.Div(
                                children=[
                                    html.Div(
                                        children=["This is where the basic settings are."],
                                        style={"marginBottom": "15px"}
                                    ),
                                    dbc.Row(
                                        children=[
                                            dbc.Col(
                                                id="basic_settings_col",
                                                children=[
                                                    dbc.Row(
                                                        children=[
                                                            dbc.Col(
                                                                parameter_input("batch_name"),
                                                                width=12
                                                            )
                                                        ],
                                                        style={"marginBottom": "10px"}
                                                    ),
                                                    dbc.Row(
                                                        children=[
                                                            dbc.Col(
                                                                parameter_input("width", step=64),
                                                                width=6
                                                            ),
                                                            dbc.Col(
                                                                parameter_input("height", step=64),
                                                                width=6
                                                            )
                                                        ],
                                                        style={"marginBottom": "10px"}
                                                    ),
                                                    dbc.Row(
                                                        children=[
                                                            dbc.Col(
                                                                parameter_input("steps"),
                                                                width=6
                                                            ),
                                                            dbc.Col(
                                                                parameter_input("n_batches"),
                                                                width=6
                                                            )
                                                        ],
                                                        style={"marginBottom": "10px"}
                                                    ),
                                                ],
                                                width=5
                                            ),
                                            dbc.Col(
                                                id="advanced_basic_settings_col",
                                                children=[
                                                    dbc.Row(
                                                        children=[
                                                            dbc.Col(
                                                                children=[
                                                                    dbc.Row(
                                                                        parameter_input("clip_guidance_scale", step=1000),
                                                                        style={"marginBottom": "10px"}
                                                                    ),
                                                                    dbc.Row(
                                                                        parameter_input("range_scale"),
                                                                        style={"marginBottom": "10px"}
                                                                    ),
                                                                    dbc.Row(
                                                                        parameter_input("sat_scale"),
                                                                        style={"marginBottom": "10px"}
                                                                    ),
                                                                    dbc.Row(
                                                                        parameter_input("tv_scale"),
                                                                        style={"marginBottom": "10px"}
                                                                    )
                                                                ],
                                                                width=5
                                                            ),
                                                            dbc.Col(
                                                                children=[
                                                                    dbc.Row(
                                                                        parameter_input("cutn_batches"),
                                                                        style={"marginBottom": "10px"}
                                                                    ),
                                                                    dbc.Row(
                                                                        children=[
                                                                            parameter_input("skip_augs", options=[
                                                                                {"label": "Skip Augs", "value": False}
                                                                            ], style={"display": "block"}),
                                                                        ],
                                                                        style={"marginBottom": "10px",
                                                                               "display": "block"}
                                                                    ),
                                                                ],
                                                                width=5
                                                            ),
                                                        ],
                                                        justify="between"
                                                    )
                                                ],
                                                width=6
                                            )
                                        ],
                                        style={"paddingLeft": "20px"},
                                        justify="between"
                                    )
                                ]
                            )
                        ])
                    ],
                    style={"marginBottom": "20px"}
                ),

                # This is the "Model Settings" row - it contains various model-selection settings
                dbc.Row(
                    id="model_settings_row",
                    children=[
                        html.Details([
                            html.Summary(
                                "Model Settings"
                            ),
                            html.Div(
                                children=["Here, you can select various model-selection settings."],
                                style={"marginBottom": "15px"}
                            ),
                            dbc.Row(
                                children=[
                                    dbc.Col(
                                        width=6,
                                        children=[
                                            dbc.Row(
                                                children=[
                                                    dbc.Col(
                                                        parameter_input("RN101", options=[{"label": ""}]),
                                                        width=12
                                                    )
                                                ],
                                                style={"marginBottom": "10px"}
                                            ),
                                            dbc.Row(
                                                children=[
                                                    dbc.Col(
                                                        parameter_input("RN50", options=[{"label": ""}]),
                                                        width=12
                                                    )
                                                ],
                                                style={"marginBottom": "10px"}
                                            ),
                                            dbc.Row(
                                                children=[
                                                    dbc.Col(
                                                        parameter_input("RN50x16", options=[{"label": ""}]),
                                                        width=12
                                                    )
                                                ],
                                                style={"marginBottom": "10px"}
                                            ),
                                            dbc.Row(
                                                children=[
                                                    dbc.Col(
                                                        parameter_input("RN50x4", options=[{"label": ""}]),
                                                        width=12
                                                    )
                                                ],
                                                style={"marginBottom": "10px"}
                                            ),
                                            dbc.Row(
                                                children=[
                                                    dbc.Col(
                                                        parameter_input("RN50x64", options=[{"label": ""}]),
                                                        width=12
                                                    )
                                                ],
                                                style={"marginBottom": "10px"}
                                            ),
                                        ]
                                    ),
                                    dbc.Col(
                                        width=6,
                                        children=[
                                            dbc.Row(
                                                children=[
                                                    dbc.Col(
                                                        parameter_input("ViTB16", options=[{"label": ""}]),
                                                        width=12
                                                    )
                                                ],
                                                style={"marginBottom": "10px"}
                                            ),
                                            dbc.Row(
                                                children=[
                                                    dbc.Col(
                                                        parameter_input("ViTB32", options=[{"label": ""}]),
                                                        width=12
                                                    )
                                                ],
                                                style={"marginBottom": "10px"}
                                            ),
                                            dbc.Row(
                                                children=[
                                                    dbc.Col(
                                                        parameter_input("ViTL14", options=[{"label": "", "value": True}]),
                                                        width=12
                                                    )
                                                ],
                                                style={"marginBottom": "10px"}
                                            ),
                                        ]
                                    )
                                ],
                                style={"paddingLeft": "20px"},
                                justify="between"

                            )
                        ])
                    ],
                    style={"marginBottom": "20px"}
                ),

                # # This is the "Animation Settings" row - it contains various settings related to animating the output
                # dbc.Row(
                #     id="animation_settings_row",
                #     children=[
                #         html.Details([
                #             html.Summary(
                #                 "Animation Settings"
                #             ),
                #             html.Div(
                #                 children=[
                #                     "Here, you'll be able to define various animation settings... once I actually "
                #                     "implement this section."],
                #                 style={"marginBottom": "10px"}
                #             ),
                #         ])
                #     ],
                #     style={"marginBottom": "20px"}
                # ),

                # This is the "Extra Settings" row - it contains various extra settings re: the model's use
                dbc.Row(
                    id="extra_settings_row",
                    children=[
                        html.Details([
                            html.Summary(
                                "Extra Settings"
                            ),
                            html.Div(
                                "This is where you can find various extra settings."
                            )
                        ])
                    ],
                    style={"marginBottom": "25px"}
                ),

                # This is the "Prompt Settings" row - it's where a user will enter their prompts
                dbc.Row(
                    children=[
                        html.Details([
                            html.Summary(
                                "Prompt Settings"
                            ),
                            html.Div(
                                children=[
                                    html.Div(
                                        children=["This is where you can define some prompts for the diffusion."],
                                        style={"marginBottom": "25px"}
                                    ),
                                    dbc.Row(
                                        children=[
                                            dbc.Col(
                                                id="prompt_control_area",
                                                children=[
                                                    dbc.Row(

                                                        # This method will generate two "Add Prompt" buttons
                                                        generate_add_prompt_buttons(),
                                                        style={"marginBottom": "25px"}
                                                    )
                                                ],
                                                width=12
                                            )
                                        ],
                                        style={"marginBottom": "10px"},
                                    )
                                ],
                                style={"width": "100%"}
                            )
                        ],
                        style={"width": "100%"})
                    ],
                    style={"marginBottom": "20px"}
                ),

                # This is the "Init Settings" row - it's where a user will define some initialization
                dbc.Row(
                    children=[
                        html.Details(
                            children=[
                                html.Summary(
                                    "Init Settings"
                                ),
                                html.Div(
                                    children=[
                                        html.Div(
                                            children=[
                                                "This is where you can define an initialization image.",
                                            ],
                                            style={"marginBottom": "25px"}
                                        ),
                                        dbc.Row(
                                            children=[
                                                dbc.Col(
                                                    children=[
                                                        dcc.Upload(
                                                            id="init_image_upload",
                                                            children=["Upload Image"],
                                                            style={"width": "734px", "height": "300px", "borderStyle": "dashed",
                                                                   "borderWidth": "1px", "textAlign": "center", "lineHeight": "300px",
                                                                   "cursor": "pointer"}
                                                        )
                                                    ],
                                                    width=8
                                                )
                                            ],
                                            justify="center",
                                            style={"marginBottom": "10px"},
                                        ),
                                        dbc.Row(
                                            children=[
                                                dbc.Col(
                                                    width=3,
                                                    children=[
                                                        html.Div(
                                                            children=[
                                                                dbc.Button(
                                                                    id="clear_init_image_button",
                                                                    children=[
                                                                        "Clear Init Image"
                                                                    ],
                                                                    style={"visibility": "hidden"}
                                                                )
                                                            ],
                                                            className="d-grid gap-2"
                                                        )
                                                    ]
                                                ),
                                            ],
                                            justify="center",
                                            style={"marginBottom": "20px", "width": "100%"},
                                        ),
                                        dbc.Row(
                                            dbc.Col(
                                                width=12,
                                                id="advanced_init_settings_col",
                                                children=[
                                                    dbc.Row(
                                                        children=[
                                                            dbc.Col(
                                                                width=5,
                                                                children=[
                                                                    parameter_input("init_scale"),
                                                                ]
                                                            ),
                                                            dbc.Col(
                                                                width=5,
                                                                children=[
                                                                    parameter_input("skip_steps"),
                                                                ]
                                                            )
                                                        ],
                                                        justify="between"
                                                    )
                                                ]
                                            )
                                        ),

                                    ],
                                ),
                            ],
                            style={"width": "100%"}
                        )
                    ],
                    style={"marginBottom": "30px"}
                ),

                # This is the "Generate Configuration File" button row.
                dbc.Row(
                    children=[
                        dbc.Row(
                            dcc.Markdown(
                                """
                                Once you're finished filling in all of the relevant settings, you can click this button to generate a configuration file! 
                                """
                            )
                        ),
                        dbc.Row(
                            children=[
                                dbc.Col(
                                    children=[
                                        dbc.Button(
                                            id="generate_config_button",
                                            color="primary",
                                            children=["Generate Config File"]
                                        )
                                    ],
                                    width=6,
                                ),
                            ],
                            style={"marginBottom": "10px"}
                        ),
                        dbc.Row(
                            children=[
                                html.Div(
                                    id="generate_config_info",
                                    children=[""]
                                )
                            ]
                        )
                    ],
                    style={"marginBottom": "30px"}
                ),

                # This is the Load Configuration File row!
                dbc.Row(
                    children=[
                        dbc.Row(
                            children=[
                                dbc.Col(
                                    width=12,
                                    children=[
                                        dcc.Markdown(
                                            f"""
                                            ### Load Configuration File
                                            If you want to slightly tweak the settings from an old file, then use this area to upload the config for a particular batch! 
                                            """
                                        )
                                    ],
                                )
                            ],
                            style={"marginBottom": "10px"}
                        ),

                        # This is the "Load Configurations" input row. Within it, there'll be an 'Upload Config File' dcc.Upload, as well as a button to clear
                        # the recently uploaded file.
                        dbc.Row(
                            children=[
                                dbc.Row(
                                    children=[
                                        dbc.Col(
                                            width=12,
                                            children=[

                                            ]
                                        )
                                    ]
                                )
                            ]
                        )
                    ],
                    style={"marginBottom": "30px"}
                ),

                # This is the 'Run Diffusion' row!
                dbc.Row(
                    children=[
                        dbc.Col(
                            children=[
                                dbc.Row(
                                    children=[
                                        dcc.Markdown(
                                            children=[
                                                """
                                                ### Run Diffusion
                                                Below, you'll be able to add configuration files to a queue; when these 
                                                files are added to the queue, they'll be automatically run... **eventually. (This option DOES NOT WORK yet.)**
                                                """
                                            ]
                                        )
                                    ]
                                ),
                                dbc.Row(
                                    children=[
                                        html.Div(
                                            children=[
                                                dcc.Upload(
                                                    id="run_diffusion_upload",
                                                    children=[
                                                        html.Div("Upload Config File")
                                                    ],
                                                    multiple=True
                                                )
                                            ],
                                            style={"width": "100%", "height": "75px", "borderStyle": "dashed",
                                                   "borderWidth": "1px", "textAlign": "center", "lineHeight": "75px",
                                                   "cursor": "pointer", "marginBottom": "20px"}
                                        )
                                    ]
                                ),
                                dbc.Row(
                                    children=[
                                        dbc.Col(
                                            id="queue_area",
                                            children=[
                                                dbc.Row(
                                                    children=[],
                                                    style={"marginBottom": "25px"}
                                                )
                                            ],
                                            width=12
                                        )
                                    ]
                                )
                            ],
                            width=12
                        )
                    ],
                    style={"marginBottom": "20px"}
                )

            ],

        )
    ],

    style={
        "marginTop": "10px",
        "marginBottom": "20px",
    }

)


# ============================
# 	   DASH APP CALLBACKS
# ============================
# Below, you'll find the callbacks that're driving this app

# This method will toggle "dev mode", which'll hide and show certain controls depending on how complicated they are
@app.callback(output=[Output("model_settings_row", "style"),
                      # Output("animation_settings_row", "style"),
                      Output("extra_settings_row", "style"),
                      Output("basic_settings_col", "width"),
                      Output("advanced_basic_settings_col", "style"),
                      Output("advanced_init_settings_col", "style")],
              inputs=[Input("dev_mode_switch", "on")])
def toggle_dev_mode_options(dev_mode_on):

    # This list will hold all of the styles
    all_styles_list = []

    # Hide / show the Rows that need to be toggled
    row_style = {"marginBottom": "25px", "display": "block"}
    if (not dev_mode_on):
        row_style["display"] = "none"
    all_styles_list = [row_style] * 2 # Change this to 3 once I reincorporate animation

    # Deal with the Basic Settings row
    basic_settings_col_width = 5 if dev_mode_on else 5
    advanced_basic_settings_style = {"display": "block" if dev_mode_on else "none"}
    all_styles_list += [basic_settings_col_width, advanced_basic_settings_style]

    # Deal with the Init Settings row
    init_settings = {"display": "block" if dev_mode_on else "none"}
    all_styles_list += [init_settings]

    # Finally, return the "all_sytles_list"
    return all_styles_list


# This (fairly lengthy) callback will manage the prompt settings space
@app.callback(output=Output("prompt_control_area", "children"),
              inputs=[Input("add_text_prompt_button", "n_clicks"),
                      Input("add_image_prompt_button", "n_clicks"),
                      Input({"type": "close_button", "index": ALL}, "n_clicks")],
              state=[State("prompt_control_area", "children")],
              prevent_initial_callback=True)
def manage_prompt_area(text_prompt_button_nclicks, image_prompt_button_nclicks,
                       prompt_close_button_nclicks,
                       prompt_area_current_children):

    global prompt_idx

    # Figure out which button was pressed
    ctx = dash.callback_context
    triggering_component = ctx.triggered[0]["prop_id"].split(".")[0]

    # If nothing was actually pressed, then we ought to not update the prompt area
    if (triggering_component == ""):
        raise PreventUpdate

    # We want to create a new list for the prompt_control_area's children, and then populate it accordingly
    prompt_area_new_children = deepcopy(prompt_area_current_children)

    # Figure out how many rows / columns there are in total
    row_ct = len(prompt_area_new_children)
    bottom_row_col_ct = len(prompt_area_new_children[-1]['props']['children'])
    total_prompts = ((row_ct - 1) * 3) + (bottom_row_col_ct - 1)

    # Here, we'll deal with the instance where triggering_component is a dict; this means that one of the "prompt close"
    # buttons triggered the callback, and we need to remove a prompt
    if (triggering_component[0] == "{"):

        # Convert the string version of the dict to an actual dict
        triggering_component = json.loads(triggering_component)
        triggering_component_id = f"prompt_col_{triggering_component['index']}"

        # Figure out which row the triggering component we're removing is in
        component_row = None
        for row_idx, row in reversed(list(enumerate(prompt_area_new_children))):
            idx_to_remove = utils.locate_id_index(triggering_component_id,
                                                  prompt_area_new_children[row_idx]['props']['children'])
            if (idx_to_remove is not None):
                component_row = row_idx
                break

        # If we're removing something from the last row, then this is simple
        if (component_row == row_ct-1):

            del prompt_area_new_children[-1]['props']['children'][idx_to_remove]

        # Otherwise, this is a little more complicated - we need to effectively "shift" all of the prompts
        # one index to the left.
        else:

            prompt_area_new_children = utils.shift_prompts_back_one(prompt_area_new_children, component_row, idx_to_remove)

    # If triggering_component isn't a dict, then we need to add a prompt
    else:

        # If we've got less than three columns in the bottom row, we can just add another column
        if (bottom_row_col_ct < 3):

            # Generate the component depending on whether or not we want a new text prompt / new image prompt
            component_to_add = generate_text_prompt_component(prompt_idx)
            if (triggering_component == "add_image_prompt_button"):
                component_to_add = generate_image_prompt_component(prompt_idx)

            # Add the component to the current prompt area
            prompt_area_new_children[-1]['props']['children'].insert(len(prompt_area_new_children[-1]['props']['children'])-1, component_to_add)

        # Otherwise, we need to add another row, and then alter things accordingly
        else:

            # Generate the component depending on whether or not we want a new text prompt / new image prompt
            component_to_add = generate_text_prompt_component(prompt_idx)
            prompt_type_to_add = "text"
            if (triggering_component == "add_image_prompt_button"):
                component_to_add = generate_image_prompt_component(prompt_idx)
                prompt_type_to_add = "image"

            # Grab the "Add Prompts" button from the list, and replace it with the component_to_add
            add_props_component = prompt_area_new_children[-1]['props']['children'][-1]
            del prompt_area_new_children[-1]['props']['children'][-1]
            prompt_area_new_children[-1]['props']['children'].append(component_to_add)

            # Now, create a new row, and then add the button to it
            prompt_area_new_children.append(dbc.Row(generate_add_prompt_buttons(), style={"marginBottom": "25px"}))

        prompt_idx += 1

    # Return the new children of the prompt_area and prompt_store
    return prompt_area_new_children


@app.callback(output=[Output("init_image_upload", "children"),
                      Output("init_image_upload", "style"),
                      Output("init_image_store", "data"),
                      Output("clear_init_image_button", "style"),
                      Output("skip_steps_input", "value")],
              inputs=[Input("init_image_upload", "filename"),
                      Input("clear_init_image_button", "n_clicks")],
              state=[State("init_image_upload", "children"),
                     State("init_image_upload", "contents"),
                     State("clear_init_image_button", "style"),
                     State("skip_steps_input", "value")])
def update_init_image_upload_area(uploaded_filename, clear_button_n_clicks, current_children, current_contents, cur_button_style,
                                  current_skip_steps_value):

    # This is the style of the Upload; we'll be returning some form of this as one of the Outputs of this callback
    upload_style = {"width": "734px", "height": "300px", "borderStyle": "dashed",
                    "borderWidth": "1px", "textAlign": "center", "lineHeight": "300px",
                    "marginTop": "3px", "cursor": "pointer"}

    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    else:
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # If the user hasn't uploaded a file, then don't change anything
    if (uploaded_filename is None):
        return current_children, upload_style, "", cur_button_style, current_skip_steps_value

    # If the user clicked the 'clear image button', we'll have to clear the area and hide the button
    elif (trigger_id == "clear_init_image_button"):

        # We'll need to change the style of the button to be hidden
        cur_button_style["visibility"] = "hidden"

        # Return everything
        return ["Upload Image"], upload_style, "", cur_button_style, 0

    # Otherwise, read in the image that the user uploaded, and then resize it so that it's displayed in the output
    else:

        # Make a new style for the "Clear Init Image" button
        cur_button_style["visibility"] = "visible"

        # We'll want to delete the lineHeight argument from the style so that the image appears in the middle of the Upload
        del upload_style["lineHeight"]

        # Load in the image
        img = utils.data_url_to_image(current_contents)

        # Save the image to the image output folder
        output_file_path = image_prompt_path / Path(uploaded_filename)
        img.save(output_file_path)

        # Resize the image, and then add it to a dcc.Graph component
        img = contain(img, (730, 298))
        fig = px.imshow(img, color_continuous_scale='gray')
        fig.update_layout(coloraxis_showscale=False)
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        fig.update_layout(margin={"l": 0, "r": 0, "t": 1, "b": 0})
        fig.update_layout(width=730)
        fig.update_layout(height=298)
        return dcc.Graph(figure=fig, config={"staticPlot": True}), upload_style, str(output_file_path), cur_button_style, 140


# This callback will update the contents of the image prompt upload box
@app.callback(output=[Output({"type": "image_prompt_upload", "index": MATCH}, "children"),
                      Output({"type": "image_prompt_upload", "index": MATCH}, "style")],
              inputs=[Input({"type": "image_prompt_upload", "index": MATCH}, "filename")],
              state=[State({"type": "image_prompt_upload", "index": MATCH}, "children"),
                     State({"type": "image_prompt_upload", "index": MATCH}, "contents")],
              prevent_initial_callback=True)
def update_image_prompt_upload_area(uploaded_filename, current_children, current_contents):

    upload_style = {"width": "100%", "height": "125px", "borderStyle": "dashed",
                    "borderWidth": "1px", "textAlign": "center", "lineHeight": "125px",
                    "cursor": "pointer"}

    if (uploaded_filename is None):
        return current_children, upload_style
    else:

        del upload_style["lineHeight"]

        # Load in the image
        img = utils.data_url_to_image(current_contents)

        # Save the image to the image output folder
        output_file_path = image_prompt_path / Path(uploaded_filename)
        img.save(output_file_path)

        # Resize the image, and then add it to a dcc.Graph component
        img = contain(img, (325, 120))
        fig = px.imshow(img, color_continuous_scale='gray')
        fig.update_layout(coloraxis_showscale=False)
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        fig.update_layout(margin={"l": 0, "r": 0, "t": 1, "b": 0})
        fig.update_layout(width=325)
        fig.update_layout(height=120)
        return dcc.Graph(figure=fig, config={"staticPlot": True}), upload_style


# This callback will update the store with all of the prompt scores
@app.callback(output=Output("prompt_score_store", "data"),
              inputs=[Input({"type": "prompt_strength", "index": ALL}, "value")],
              state=[State("prompt_score_store", "data"),
                     State({"type": "prompt_strength", "index": ALL}, "value")])
def update_prompt_score_store(current_prompt_strengths, current_prompt_score_store_data,
                              stored_prompt_strengths):

    # Figure out which prompt's strength was changed
    ctx = dash.callback_context

    # We're going to create a "new" version of the current_prompt_score_store_data
    new_prompt_score_store_data = deepcopy(current_prompt_score_store_data)

    # If triggering_component has more than one component in it, then we'll wipe out the current
    # dictionary and replace it with a new one.
    if (len(ctx.triggered) > 1 or len(stored_prompt_strengths)==1):
        new_prompt_score_store_data = {}

    # Iterate through each of the components and update the new store data accordingly
    for component in ctx.triggered:

        # If nothing triggered this, then raise a Prevent Update
        if (component["prop_id"] == "."):
            raise PreventUpdate

        # Grab some information about this component
        component_info = json.loads(component["prop_id"][:-6])
        component_value = component["value"]

        # Update the store accordingly
        new_prompt_score_store_data[str(component_info["index"])] = component_value

    return new_prompt_score_store_data


# This callback will update the prompt influence bars according to the value in the prompt score store
@app.callback(output=Output({"type": "prompt_influence", "index": ALL}, "value"),
              inputs=[Input("prompt_score_store", "data")])
def update_prompt_influence_bars(prompt_score_store):

    # Calculate the total prompt score
    total_prompt_score = sum([abs(val) for key, val in prompt_score_store.items()])

    # If all of the sliders are set to 0, then return an array of all 0's
    if (total_prompt_score == 0):
        return [0] * len(prompt_score_store)

    # Calculate all of the "slider values"
    slider_values = [100*(abs(val)/total_prompt_score) for key, val in prompt_score_store.items()]

    # Return the slider values
    return slider_values


# This callback will update the "prompt" store with any props in the input boxes
@app.callback(output=Output("prompt_store", "data"),
              inputs=[Input({"type": "image_prompt_upload", "index": ALL}, "filename"),
                      Input({"type": "text_prompt_input", "index": ALL}, "value")],
              state=[State("prompt_store", "data")])
def update_prompt_store(image_prompt_filenames, text_prompt_inputs, current_prompt_store_data):

    # Figure out what triggered this callback
    ctx = dash.callback_context

    # We're going to create a "new" version of the current_prompt_store_data
    new_prompt_store_data = deepcopy(current_prompt_store_data)

    # If triggering_component has more than one component in it, then we'll wipe out the current
    # dictionary and replace it with a new one.
    if (len(ctx.triggered) > 1):
        new_prompt_store_data = {}

    # Iterate through each of the components and update the new store data accordingly
    for component in ctx.triggered:

        # If nothing triggered this, then raise a Prevent Update
        if (component["prop_id"] == "."):
            raise PreventUpdate

        # Grab some information about this component
        if ("image" in component["prop_id"]):
            component_info = json.loads(component["prop_id"][:-9])
            prompt_type = "image"
        else:
            component_info = json.loads(component["prop_id"][:-6])
            prompt_type = "text"
        component_value = component["value"]

        # If the value isn't None, then add it to the dict
        if component_value is not None:
            new_prompt_store_data[str(component_info["index"])] = {"type": prompt_type, "value": component_value}

    # Now, return the entire new_prompt_store_data
    return new_prompt_store_data


# This callback will launch the 'generate configuration' process, which will create a config file
@app.callback(output=Output("generate_config_info", "children"),
              inputs=[Input("generate_config_button", "n_clicks")],
              state=[State("prompt_store", "data"),
                     State("prompt_score_store", "data"),
                     State("settings_store", "data"),
                     State("init_image_store", "data")])
def generate_config(generate_config_nclicks,
                    prompt_store_data, prompt_score_store_data, settings_store_data, init_image_store_data):

    # If there haven't been any clicks, then we shouldn't run this
    if (generate_config_nclicks is None):
        raise PreventUpdate

    # If there's an init image, then we'll want to edit the settings_store_data
    if (init_image_store_data != "" and init_image_store_data is not None):
        settings_store_data["init_image"] = init_image_store_data
        skip_amt = 140
        settings_store_data["skip_steps"] = skip_amt
        settings_store_data["steps"] += skip_amt

    # Run the "generate config" method in utils
    utils.generate_config(prompt_store_data, prompt_score_store_data, settings_store_data)

    # Return a message for the generate_config_info Div
    saved_filename = "/config_files/"
    if ("batch_name" in settings_store_data):
        saved_filename = f"{saved_filename}{settings_store_data['batch_name']}"
    return dcc.Markdown(
        f"""Generated a new configuration file at **{saved_filename}.**"""
    )

    raise PreventUpdate


# This callback will update the settings_store based on certain controls
@app.callback(output=Output("settings_store", "data"),
              inputs=[param_settings_inputs],
              state=State("settings_store", "data"))
def update_settings_store(inputs, current_settings):

    # Figure out which button was pressed
    ctx = dash.callback_context
    triggering_component = ctx.triggered[0]["prop_id"].split(".")[0]
    param_name = "_".join(triggering_component.split("_")[:-1])

    # If nothing was
    if (triggering_component == ""):
        raise PreventUpdate

    # Create a 'new settings' dict
    new_settings = deepcopy(current_settings)

    # Use the component_to_settings dict to figure out what setting we need to update
    value = ctx.triggered[0]["value"]
    new_settings[param_name] = value

    return new_settings


# This (fairly lengthy) callback will manage the prompt settings space
@app.callback(output=Output("queue_area", "children"),
              inputs=[Input("run_diffusion_upload", "contents"),
                      Input({"type": "config_close_button", "index": ALL}, "n_clicks")],
              state=[State("queue_area", "children")],
              prevent_initial_callback=True)
def manage_config_queue(upload_contents, config_close_button_nclicks,
                        queue_area_current_children):

    global config_idx
    global config_queue
    global config_queue_status

    # We're going to keep a list of the configs that were uploaded
    uploaded_configs = {}

    # Figure out which button was pressed
    ctx = dash.callback_context
    triggering_component = ctx.triggered[0]["prop_id"].split(".")[0]

    # If nothing was actually pressed, then we ought to not update the prompt area
    if (triggering_component == ""):
        raise PreventUpdate

    # We want to create a new list for the queue_area's children, and then populate it accordingly
    queue_area_new_children = deepcopy(queue_area_current_children)

    # Figure out how many rows / columns there are in total
    row_ct = len(queue_area_new_children)
    bottom_row_col_ct = len(queue_area_new_children[-1]['props']['children'])
    total_configs = max(((row_ct - 1) * 3) + (bottom_row_col_ct), 0)

    # Deal with the instance where we'll need to remove a config
    if (triggering_component[0] == "{"):

        # Convert the string version of the dict to an actual dict
        triggering_component = json.loads(triggering_component)
        triggering_component_id = f"config_col_{triggering_component['index']}"

        # Figure out which row the triggering component we're removing is in
        component_row = None
        for row_idx, row in reversed(list(enumerate(queue_area_new_children))):
            idx_to_remove = utils.locate_id_index(triggering_component_id,
                                                  queue_area_new_children[row_idx]['props']['children'])
            if (idx_to_remove is not None):
                component_row = row_idx
                break

        # If we're removing something from the last row, then this is simple
        if (component_row == row_ct-1):

            del queue_area_new_children[-1]['props']['children'][idx_to_remove]

        # Otherwise, this is a little more complicated - we need to effectively "shift" all of the prompts
        # one index to the left.
        else:
            queue_area_new_children = utils.shift_prompts_back_one(queue_area_new_children, component_row, idx_to_remove)

        del config_queue[triggering_component['index']]
        config_queue_status[triggering_component['index']] = {"status": "stopped", "progress": config_queue_status[triggering_component['index']]["progress"]}

    # If triggering_component isn't a dict, then we need to add a config
    else:

        # Since we can upload multiple configuration files at once, we'll want to make sure that we're adding
        # windows to handle all of them
        for upload in upload_contents:

            # Recalculate some of this information
            row_ct = len(queue_area_new_children)
            bottom_row_col_ct = len(queue_area_new_children[-1]['props']['children'])
            total_configs = max(((row_ct - 1) * 3) + (bottom_row_col_ct), 0)

            # If we've got less than three columns in the bottom row, we can just add another column
            if (bottom_row_col_ct < 3):

                # Generate the component depending on whether or not we want a new text prompt / new image prompt
                component_to_add = generate_config_window(config_idx)

                # Add the component to the current prompt area
                queue_area_new_children[-1]['props']['children'].append(component_to_add)

            # Otherwise, we need to add another row, and then alter things accordingly
            else:

                # Generate the component depending on whether or not we want a new text prompt / new image prompt
                component_to_add = generate_config_window(config_idx)

                # Now, create a new row, and then add the button to it
                queue_area_new_children.append(dbc.Row(children=[component_to_add], style={"marginBottom": "25px"}))

            # Parse the contents of the uploaded configurations, and add them to the uploaded_configs list
            uploaded_configs[config_idx] = utils.parse_uploaded_config(upload)
            config_queue_status[config_idx] = {"status": "queued", "progress": 0}

            # Update the config_idx
            config_idx += 1

        # Add the contents of the uploaded_configs list to the config_queue_data store
        config_queue = config_queue | uploaded_configs

    # Return the new children of the prompt_area and prompt_store
    return queue_area_new_children


# This callback will print how many intervals have elapsed
@app.callback(output=Output({"type": "config_progress_bar", "index": ALL}, "value"),
              inputs=[Input("queue_management_interval", "n_intervals")],
              state=[State({"type": "config_progress_bar", "index": ALL}, "value")])
def handle_queue(n_intervals, cur_vals):

    # First, we're going to check to see if there are un-run config files in the queue
    for config_idx, config_status in config_queue_status.items():
        if (config_status["status"] == "queued"):
            print(f"At timestep {n_intervals}, config {config_idx} needs to be run.")

    raise PreventUpdate


# ============================
# 	        MAIN
# ============================
# Below, you'll find the code that actually launches the application

# This callback will update the prompt_score_store with the most current information
app.run_server(debug=True, port=4200)
