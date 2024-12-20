# Movie Recommender System

```
 THIS IS A WORK IN PROGRESS
 ```

## Description
This project is a movie recommendation system that predicts what users will rate movies they haven't watched yet within the train set and recommends 10 movies based on what the algorithm thinks they will like the most. Explains why recommendations are made and is transparent about latent feature weighting. Gives evaluation metrics for recommendation assessment.

Output includes:
- top 10 recommendations list with movie ids
- top 10 recommendations list with movie names
- the contributions of similar users, genres, and tag weightings
- explanations for why the top 3 movies were recommended to add transparency to recommendations

## Features
- Hybrid filter approach to take advantage of similar users as well as user metadata
- Collaborative filtering through SVD
- Content-based filtering through Word2Vec neural network to apply natural language processing to analyze movie genres and user-submitted review tags
- Regression 
- Recommendation explainability and transparency

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/tylerho5/school-projects/movie-recommender-system.git
    ```

2. Create a virtual environment and install dependencies:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

## Usage
1. Place your datasets in the `datasets/` folder.
2. Run the main script:
    ```bash
    python recSys_final.py
    ```
3. Outputs will be saved in the `output/` directory

## Folder Structure
```markdown
├── output/             # Contains recommendation results
├── raw-datasets/       # Contains raw datasets
├── scripts/            # Python scripts for recommender system
├── requirements.txt    # List of dependencies
└── README.md           # Project documentation
```

## Acknowledgments

This project was built with inspiration and code references from the following open-source repositories:

- **scikit-learn** (BSD 3-Clause License)
  - [Truncated Singular Value Decomposition (SVD)](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/decomposition/_truncated_svd.py) - Referenced for implementing [`SVD.py`](./scripts/SVD.py).
  - [Ridge Regression](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/linear_model/_ridge.py) - Used as a reference for [`RegressionModel.py`](./scripts/RegressionModel.py).
  - [Preprocessing Scaler](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/preprocessing/_data.py) - Consulted for [`Scaler.py`](./scripts/Scaler.py).

- **TensorFlow** (Apache 2.0 License)
  - [Word2Vec Tutorial](https://github.com/tensorflow/text/blob/master/docs/tutorials/word2vec.ipynb) - Used for implementing [`Word2Vec.py`](./scripts/Word2Vec.py).

- **Gensim** (LGPL License)
  - [Word2Vec Implementation](https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/models/word2vec.py) - Referenced for enhancing the Word2Vec model.

We are grateful to the open-source community for these resources, which have significantly contributed to the inspiration and development of the project.

## Contributors

This project was a collaborative effort between:
- Tyler Ho: Project lead, implementation of custom SVD, custom Word2Vec, custom scaler, custom regression, as well as documentation and codebase organiziation.
- [Quynh Nguyen](https://www.linkedin.com/in/quynhnng/): Contributions in research, testing, technical report, and presentation.

## License

This project is licensed under the BSD 3-Clause License. See the [LICENSE](./LICENSE) file in this repository for details.

