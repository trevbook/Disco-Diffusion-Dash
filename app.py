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

# Dash-related imports
import dash
from dash.dependencies import Input, Output, State
from dash import dash_table
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

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
# 	    LOADING DATA
# ============================
# The code below will load a couple of data structures into RAM; these will be used
# as the initial items for the various data stores


# ============================
# 	    GENERAL METHODS
# ============================
# All of the code below are more general methods I can use throughout
# the rest of this app's code-base


# ============================
# 	   DATA MANIPULATION
# ============================
# The code below is meant to maniuplate my data in whatever way
# is necessary for me to display the right data that I want


# ============================
# 	    DASH APP LAYOUT
# ============================
# Below, you'll find code that's directly defining the app's layout

# Setting up the main layout of the Dash app
app.layout = html.Div(

   children=[

      # This is the container for the entire Dash app
      dbc.Container(

         children=[

            # This is the "header" row - it contains some introductory information about the app itself
            dbc.Row(
               dcc.Markdown(
                  children=[
                     """
                     # Disco Diffusion Dash
                     This is a local version of [Disco Diffusion v4.1](https://colab.research.google.com/drive/1sHfRn5Y0YKYKi1k-ifUSBFRNJ8_1sa39?usp=sharing#scrollTo=LHLiO56OfwgD).
                     This UI was written by Trevor Hubbard using [Plotly Dash](https://plotly.com/dash/). 
                     """
                  ]
               ),
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
                                          ]
                                       )
                                    ],
                                    width=5
                                 ),
                                 dbc.Col(
                                    children=[
                                       dbc.Row(
                                          children=[
                                             dbc.Col(
                                                children=[
                                                   dbc.Row(
                                                      children=[
                                                         dbc.Label(html.B("Steps")),
                                                         dbc.Input(id="steps_input")
                                                      ],
                                                      style={"marginBottom": "10px"}
                                                   ),
                                                   dbc.Row(
                                                      children=[
                                                         dbc.Label(html.B("CutN Batches")),
                                                         dbc.Input(id="cutn_batches_input")
                                                      ],
                                                      style={"marginBottom": "10px"}
                                                   ),
                                                   dbc.Row(
                                                      children=[
                                                         dbc.Label(html.B("Augmentations")),
                                                         dcc.Checklist(
                                                            id="skip_augs_checkbox",
                                                            options=[{"label": "Skip Augs", "value": "skip_augs"}],
                                                            value=[]
                                                         )
                                                      ],
                                                      style={"marginBottom": "10px"}
                                                   ),
                                                ],
                                                width=5
                                             ),
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
                                             )
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
               children=[
                  html.Details([
                     html.Summary(
                        "Animation Settings"
                     ),
                     html.Div(
                        children=["Here, you'll be able to define various animation settings... once I actually "
                                  "implement this section."],
                        style={"marginBottom": "10px"}
                     ),
                  ])
               ],
               style={"marginBottom": "20px"}
            ),

            # This is the "Extra Settings" row - it contains various extra settings re: the model's use
            dbc.Row(
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
               style={"marginBottom": "20px"}
            ),

            # This is the "Prompt Settings" row - it's where a user will enter their prompts
            dbc.Row(
               children=[
                  html.Details([
                     html.Summary(
                        "Prompt Settings"
                     ),
                     html.Div(
                        "This is where you can define some prompts for the diffusion."
                     )
                  ])
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

         ]

      )
   ],

   style={
      "marginTop": "10px",
      "marginBottom": "20px"
   }

)

# ============================
# 	   DASH APP CALLBACKS
# ============================
# Below, you'll find the callbacks that're driving this app

# @app.callback(output=[Output()],
#               inputs=[])


# ============================
# 	        MAIN
# ============================
# Below, you'll find the code that actually launches the application

app.run_server(debug=True, port=4200)
