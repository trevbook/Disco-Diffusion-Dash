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


# This method will generate a 'config_window'
def generate_config_window(config_num):

    component = dbc.Col(
        id=f"config_col_{config_num}",
        children=[
            html.Div(
                children=[

                    # This is the "header row" of the config window
                    dbc.Row(
                        children=[
                            dbc.Col(
                                children=[
                                    html.B(f"Config Window {config_num}")
                                ],
                                width=10
                            ),
                            dbc.Col(
                                children=[
                                    dbc.Button(
                                        id={"type": "config_close_button", "index": config_num},
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

                    # This is the Progress Bar row
                    dbc.Row(
                        children=[
                            dbc.Col(
                                children=[
                                    dbc.Progress(id={"type": "config_progress_bar", "index": config_num})
                                ]
                            )
                        ]
                    ),

                ],
                style={"border": "1px solid green", "height": "300px", "padding": "10px"},
            )
        ],
        width=4
    )

    return component


