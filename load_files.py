import os
import json


def list_json_files(directory):
    try:
        # List all files in the given directory
        files = os.listdir(directory)

        # Filter out and return only the JSON files
        json_files = [file for file in files if file.endswith(".json")]
        return json_files
    except FileNotFoundError:
        print(f"The directory {directory} does not exist.")
        return []


def load_json(file_path):
    with open(file_path, "r") as file:
        return json.load(file)


def load_all_files():
    while True:
        file_name = input(
            "Please enter the name of the file containing the texts (press Enter to use 'app1.2.json'): "
        )
        if not file_name:
            file_name = "app1.2.json"
        if os.path.isfile(file_name):
            break  # Exit the loop if the file is successfully read
        else:
            print(
                f"File '{file_name}' not found. Please make sure the file exists and try again."
            )
    json_files = ["app1.json", "app2.json"]
    json_objects = [load_json(file) for file in json_files]
    query_json = load_json(file_name)
    return json_objects, query_json
