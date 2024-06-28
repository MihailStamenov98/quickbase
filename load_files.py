import os
import json


def list_json_files(directory):
    try:
        files = os.listdir(directory)
        json_files = [file for file in files if file.endswith(".json")]
        return json_files
    except FileNotFoundError:
        print(f"The directory {directory} does not exist.")
        return []


def load_json(file_path):
    with open(file_path, "r") as file:
        return json.load(file)


def load_all_files(directory="data/data_pool/"):
    while True:
        file_name = input(
            "Please enter the name of the file containing the texts (press Enter to use 'app1.2.json'): "
        )
        if not file_name:
            file_name = "data/query_data/app1.2.json"
        if os.path.isfile(file_name):
            break  # Exit the loop if the file is successfully read
        else:
            print(
                f"File '{file_name}' not found. Please make sure the file exists and try again."
            )
    json_files = list_json_files(directory=directory)
    prefixed_files = [f"{directory}{file}" for file in json_files]
    json_objects = [load_json(file) for file in prefixed_files]
    query_json = load_json(file_name)
    return [json_objects[0]], query_json
