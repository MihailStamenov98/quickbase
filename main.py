import time
from predictors import GraphSimilarity, TF_IDF, Encoders
from load_files import load_all_files


def main():
    start_time = time.time()

    print("I have 3 solutions:")
    print(
        "1. Encoding with BERT (If you chooe that the first time it needs to download the models. Once the models are downloaded it works.)"
    )
    print(
        "2. Encoding with Electra (If you chooe that the first time it needs to download the models. Once the models are downloaded it works.)"
    )
    print("3. Encoding with TF-IDF")
    print("4. Graph similarity")

    choice = input("Please enter the number of the solution you want to use: ")

    while choice not in {"1", "2", "3", "4"}:
        print("Invalid choice. Please enter 1, 2, or 3.")
        choice = input("Please enter the number of the solution you want to use: ")

    json_objects, query_json = load_all_files()
    model = None
    if choice == "1":
        print("You chose encoding with BERT.")
        model = Encoders("bert")
    elif choice == "2":
        print("You chose encoding with Electra.")
        model = Encoders("electra")
    elif choice == "3":
        print("You chose encoding with TF-IDF.")
        model = TF_IDF(json_objects)
    elif choice == "4":
        print("You chose Graph similarity.")
        model = GraphSimilarity(json_objects=json_objects)
    else:
        print("Invalid choice. Please enter 1, 2, or 3.")
        return
    most_similar = model.predict(query_json=query_json)
    print(most_similar)
    print("Elapsed time: ", time.time() - start_time)


if __name__ == "__main__":
    main()
