import pandas as pd
import json

json_file_path = 'results_i5-4670K.json'

with open(json_file_path, 'r') as file:
    data = json.load(file)

df = pd.DataFrame(data)

csv_file_path = 'results_i5-4670K.csv'
df.to_csv(csv_file_path, index=False)

print(f'DataFrame saved to {csv_file_path}')