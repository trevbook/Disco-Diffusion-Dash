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
from utils import generate_text_prompt_component, generate_image_prompt_component, generate_add_prompt_buttons

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
        dcc.Store(id="prompt_store", data=[]),
        dcc.Store("misc_store"),

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
                              """
                                    ]
                                ),
                            ],
                            width=8,
                        ),
                        dbc.Col(
                            children=[
                                daq.BooleanSwitch(
                                    id="dev_mode_switch",
                                    on=False,
                                    label="Developer Mode",
                                    labelPosition="top"
                                )
                            ],
                            width=4
                        )
                    ],
                    style={"marginBottom": "20px"}
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
                                                                children=[
                                                                    dbc.Label(html.B("Batch Name")),
                                                                    dbc.Input(id="batch_name_input")
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
                                                                    dbc.Label(html.B("Image Width")),
                                                                    dbc.Input(id="width_input")
                                                                ],
                                                                width=6
                                                            ),
                                                            dbc.Col(
                                                                children=[
                                                                    dbc.Label(html.B("Image Height")),
                                                                    dbc.Input(id="height_input")
                                                                ],
                                                                width=6
                                                            )
                                                        ],
                                                        style={"marginBottom": "10px"}
                                                    ),
                                                    dbc.Row(
                                                        children=[
                                                            dbc.Col(
                                                                children=[
                                                                    dbc.Label(html.B("Steps")),
                                                                    dbc.Input(id="steps_input")
                                                                ],
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
                                                                        children=[
                                                                            dbc.Label(html.B("CLIP Guidance Scale")),
                                                                            dbc.Input(id="clip_guidance_scale_input")
                                                                        ],
                                                                        style={"marginBottom": "10px"}
                                                                    ),
                                                                    dbc.Row(
                                                                        children=[
                                                                            dbc.Label(html.B("Range Scale")),
                                                                            dbc.Input(id="range_scale_input")
                                                                        ],
                                                                        style={"marginBottom": "10px"}
                                                                    ),
                                                                    dbc.Row(
                                                                        children=[
                                                                            dbc.Label(html.B("Sat Scale")),
                                                                            dbc.Input(id="sat_scale_input")
                                                                        ],
                                                                        style={"marginBottom": "10px"}
                                                                    ),
                                                                    dbc.Row(
                                                                        children=[
                                                                            dbc.Label(html.B("TV Scale")),
                                                                            dbc.Input(id="tv_scale_input")
                                                                        ],
                                                                        style={"marginBottom": "10px"}
                                                                    )
                                                                ],
                                                                width=5
                                                            ),
                                                            dbc.Col(
                                                                children=[
                                                                    dbc.Row(
                                                                        children=[
                                                                            dbc.Label(html.B("CutN Batches")),
                                                                            dbc.Input(id="cutn_batches_input")
                                                                        ],
                                                                        style={"marginBottom": "10px"}
                                                                    ),
                                                                    dbc.Row(
                                                                        children=[
                                                                            dbc.Label(
                                                                                children=[html.B("Augmentations")],
                                                                                style={"display": "block"}
                                                                            ),
                                                                            dcc.Checklist(
                                                                                id="skip_augs_checkbox",
                                                                                options=[{"label": "   Skip Augs",
                                                                                          "value": "skip_augs"}],
                                                                                value=[],
                                                                                style={"display": "block"}
                                                                            )
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
                                style={"marginBottom": "10px"}
                            )
                        ])
                    ],
                    style={"marginBottom": "20px"}
                ),

                # This is the "Animation Settings" row - it contains various settings related to animating the output
                dbc.Row(
                    id="animation_settings_row",
                    children=[
                        html.Details([
                            html.Summary(
                                "Animation Settings"
                            ),
                            html.Div(
                                children=[
                                    "Here, you'll be able to define various animation settings... once I actually "
                                    "implement this section."],
                                style={"marginBottom": "10px"}
                            ),
                        ])
                    ],
                    style={"marginBottom": "20px"}
                ),

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
                        html.Details([
                            html.Summary(
                                "Init Settings"
                            ),
                            html.Div(
                                "This is where you can define an initialization image."
                            )
                        ])
                    ],
                    style={"marginBottom": "20px"}
                ),

                # This is the "Diffusion Settings" row - it's where users will define some settings for running diffusion
                dbc.Row(
                    children=[
                        html.Details([
                            html.Summary(
                                "Diffusion Settings"
                            ),
                            html.Div(
                                "This is where you can run diffusion!"
                            )
                        ])
                    ],
                    style={"marginBottom": "20px"}
                ),

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
                      Output("animation_settings_row", "style"),
                      Output("extra_settings_row", "style"),
                      Output("basic_settings_col", "width"),
                      Output("advanced_basic_settings_col", "style")],
              inputs=[Input("dev_mode_switch", "on")])
def toggle_dev_mode_options(dev_mode_on):

    # This list will hold all of the styles
    all_styles_list = []

    # Hide / show the Rows that need to be toggled
    row_style = {"marginBottom": "25px", "display": "block"}
    if (not dev_mode_on):
        row_style["display"] = "none"
    all_styles_list = [row_style] * 3

    # Deal with the Basic Settings row
    basic_settings_col_width = 5 if dev_mode_on else 12
    advanced_basic_settings_style = {"display": "block" if dev_mode_on else "none"}
    all_styles_list += [basic_settings_col_width, advanced_basic_settings_style]

    # Finally, return the "all_sytles_list"
    return all_styles_list

@app.callback(output=Output("misc_store", "data"),
              inputs=[Input("prompt_store", "data")],
              prevent_initial_callback=True)
def print_cur_prompt_store(prompt_store_data):
    return prompt_store_data


# This (fairly lengthy) callback will manage the prompt settings space
@app.callback(output=[Output("prompt_control_area", "children"),
                      Output("prompt_store", "data")],
              inputs=[Input("add_text_prompt_button", "n_clicks"),
                      Input("add_image_prompt_button", "n_clicks"),
                      Input({"type": "close_button", "index": ALL}, "n_clicks")],
              state=[State("prompt_control_area", "children"),
                     State("prompt_store", "data")],
              prevent_initial_callback=True)
def manage_prompt_area(text_prompt_button_nclicks, image_prompt_button_nclicks,
                       prompt_close_button_nclicks,
                       prompt_area_current_children, prompt_store_current_data):

    global prompt_idx

    # Figure out which button was pressed
    ctx = dash.callback_context
    triggering_component = ctx.triggered[0]["prop_id"].split(".")[0]

    # If nothing was actually pressed, then we ought to not update the prompt area
    if (triggering_component == ""):
        raise PreventUpdate

    # We want to create a new list for the prompt_control_area's children, and then populate it accordingly
    prompt_area_new_children = deepcopy(prompt_area_current_children)

    # We also want to populate the "prompt_store_data", which will contain a list of which prompt types we have
    prompt_store_new_data = deepcopy(prompt_store_current_data)

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
            prompt_type_to_add = "text"
            if (triggering_component == "add_image_prompt_button"):
                component_to_add = generate_image_prompt_component(prompt_idx)
                prompt_type_to_add = "image"

            # Add the component to the current prompt area
            prompt_area_new_children[-1]['props']['children'].insert(len(prompt_area_new_children[-1]['props']['children'])-1, component_to_add)

            # Add the prompt type to the prompt_store
            prompt_store_new_data.append(prompt_type_to_add)

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
    return [prompt_area_new_children, prompt_store_new_data]


# This callback will update the contents of the image prompt upload box
@app.callback(output=[Output({"type": "image_prompt_upload", "index": MATCH}, "children"),
                      Output({"type": "image_prompt_upload", "index": MATCH}, "style")],
              inputs=[Input({"type": "image_prompt_upload", "index": MATCH}, "filename")],
              state=[State({"type": "image_prompt_upload", "index": MATCH}, "children"),
                     State({"type": "image_prompt_upload", "index": MATCH}, "contents")],
              prevent_initial_callback=True)
def update_image_prompt_upload_text(uploaded_filename, current_children, current_contents):


    upload_style = {"width": "100%", "height": "125px", "borderStyle": "dashed",
                    "borderWidth": "1px", "textAlign": "center", "lineHeight": "125px",
                    "cursor": "pointer"}

    if (uploaded_filename is None):
        return current_children, upload_style
    else:

        del upload_style["lineHeight"]

        img = utils.data_url_to_image(current_contents)
        img = contain(img, (325, 120))
        fig = px.imshow(img, color_continuous_scale='gray')
        fig.update_layout(coloraxis_showscale=False)
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        fig.update_layout(margin={"l": 0, "r": 0, "t": 1, "b": 0})
        fig.update_layout(width=325)
        fig.update_layout(height=120)
        return dcc.Graph(figure=fig, config={"staticPlot": True}), upload_style



# ============================
# 	        MAIN
# ============================
# Below, you'll find the code that actually launches the application

app.run_server(debug=True, port=4200)
