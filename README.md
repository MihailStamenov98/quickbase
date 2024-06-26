# Database Schema Similarity Solutions

This project intends to give example solutions to the problem described in `task.md`.

## Overview

To test the solutions, run `main.py` and follow the instructions. Three approaches were implemented:

1. **BERT Encoder**:
   - Descriptions of all given JSON files are encoded using the BERT encoder to receive vector representations of all JSON files.
   - Given a new JSON, the same procedures are applied to the query JSON, and its cosine similarities with all embeddings are calculated.
   - The highest similarity is taken.

2. **ELECTRA Encoder**:
   - Uses the same idea as the BERT Encoder approach but employs the ELECTRA encoder instead of BERT.

3. **TF-IDF Vectorization**:
   - Creates vectors of all TF-IDF values of all words from the dictionary created from the words in the JSON files.

## Execution Notes

- The execution of the first two solutions requires the BERT or ELECTRA models to be downloaded locally for faster execution. If the user does not have them, run `main.py` and try using the chosen solutions. The code will execute, download the models, but will finish with an error. After that, the executions of the respective solutions should be available.

- The code uses libraries such as `transformers` and `torch`, and the models are configured to run on the GPU for faster execution. Therefore, the user should have PyTorch with CUDA installed and configured.

## Requirements

- Python 3.x
- PyTorch with CUDA (for GPU acceleration)
- `transformers` library
- `scikit-learn` library

## Installation

1. Clone the repository:
   git clone https://github.com/yourusername/yourproject.git
   cd yourproject

## Usage

1. Ensure you have the BERT or ELECTRA models downloaded locally for faster execution. If not, the code will attempt to download the models when you run the solutions, which may result in an error if the download fails. After the initial attempt, the models should be available for subsequent runs.

2. Run `main.py`:
   python main.py

3. Follow the instructions provided by `main.py` to test the solutions.

## Libraries Used

- `transformers`: For BERT and ELECTRA models.
- `torch`: For tensor operations and GPU acceleration.
- `scikit-learn`: For TF-IDF vectorization and cosine similarity calculations.

## Contact

For any issues or questions, please contact [mpstamenov@gmail.com].
