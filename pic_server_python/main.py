from figma_function import figma_fn
import json


with open('./figma_data/raw/Android_56_4.json') as f:
    json_data = json.load(f)

ui_name = 'Android_56'
print(figma_fn(json_data, ui_name))