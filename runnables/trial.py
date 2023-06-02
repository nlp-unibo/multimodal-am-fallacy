import pandas as pd
import os

# read the predictions file in leave_one_out/text_audio/run_18/predictions.json 
project_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
pred_file_json = os.path.join(project_dir, 'results', 'leave_one_out', 'text_audio', 'run_18', 'predictions.json')
pred_file_csv = os.path.join(project_dir, 'results', 'leave_one_out', 'text_audio', 'run_18', 'predictions.csv')

# read the predictions file
df_pred = pd.read_json(pred_file_json, orient='index')
print(df_pred.head())

df_pred = pd.read_csv(pred_file_csv)
print(df_pred.head())

