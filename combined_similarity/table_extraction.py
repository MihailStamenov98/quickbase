import json

def extract_schema(json_data):
    schema = {}
    for app_id, app_content in json_data.items():
        tables = app_content['ai_dict']['tables']
        schema[app_id] = {table['name']: table for table in tables}
    return schema

with open('/mnt/data/data.json') as f:
    data = json.load(f)

schemas = extract_schema(data)
