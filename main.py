import time
from functools import partial
from json_to_schema_description import *
from utils import *
from encoders import *
from TfidfVectorizer_encoder import *

def main():
    start_time = time.time()

    print("I have 3 solutions:")
    print("1. Encoding with BERT (If you chooe that the first time it needs to download the models. Once the models are downloaded it works.)")
    print("2. Encoding with Electra (If you chooe that the first time it needs to download the models. Once the models are downloaded it works.)")
    print("3. Encoding with TF-IDF")
    
    choice = input("Please enter the number of the solution you want to use: ")

    while choice not in {'1', '2', '3'}:
        print("Invalid choice. Please enter 1, 2, or 3.")
        choice = input("Please enter the number of the solution you want to use: ")

    while True:
        file_name = input("Please enter the name of the file containing the texts (press Enter to use 'app1.2.json'): ")
        if not file_name:
            file_name = 'app1.2.json'
        if os.path.isfile(file_name):
            with open(file_name, 'r') as file:
                texts = file.readlines()
                texts = [line.strip() for line in texts]
            break  # Exit the loop if the file is successfully read
        else:
            print(f"File '{file_name}' not found. Please make sure the file exists and try again.")
           
    json_files = ["app1.json", "app2.json"]
    json_objects = [load_json(file) for file in json_files]
    query_json = load_json(file_name)
    start_time = time.time()

    if choice == '1':
        print("You chose encoding with BERT.")
        most_similar = get_most_similar(json_objects=json_objects, 
                                query_json=query_json, 
                                feature_extraction_func=partial(transformer_feature_extractor, 
                                                                model='bert'), 
                                embedings_path='bert_encodings')
    elif choice == '2':
        print("You chose encoding with Electra.")
        most_similar = get_most_similar(json_objects=json_objects, 
                                query_json=query_json, 
                                feature_extraction_func=partial(transformer_feature_extractor, 
                                                                model='electra'),
                                embedings_path='electra_encodings')
    elif choice == '3':
        print("You chose encoding with TF-IDF.")
        most_similar = tf_idf_encoding_compare(json_objects=json_objects, 
                                                query_json=query_json)
    else:
        print("Invalid choice. Please enter 1, 2, or 3.")
    end_time = time.time()

    print(most_similar.get("ai_dict", {}).get("name", "Unknown App"))
    print("Elapsed time: ", end_time-start_time)

if __name__ == "__main__":
    main()