import pandas as pd
import json

params_df = pd.read_excel("Parameter Explanations.xlsx")
important_params_df = params_df.query("type=='str' | type=='int' | type=='bool' | type=='float'")
param_defaults_dict = {row["parameter_name"]:row["default_colab_value"] for idx, row in important_params_df.iterrows()}
with open("param_defaults.json", "w") as param_json:
	json.dump(param_defaults_dict, param_json, indent=4)