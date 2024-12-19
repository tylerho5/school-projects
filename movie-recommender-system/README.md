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

This project was built with inspiration and code references from the following open-source repositories. We have much appreciation for the contributions of the developers and maintainers of these projects.

- **[scikit-learn](https://github.com/scikit-learn/scikit-learn)**:
  - [Truncated Singular Value Decomposition (SVD)](https://github.com/scikit-learn/scikit-learn/blob/fa5d7275b/sklearn/decomposition/_truncated_svd.py#L28) - Used as a reference for [our simplified implementation of SVD](https://github.com/tylerho5/school-projects/blob/main/movie-recommender-system/scripts/SVD.py).
  - [Ridge Regression](https://github.com/scikit-learn/scikit-learn/blob/6e9039160/sklearn/linear_model/_ridge.py#L1016) - Referred for insights on [implementing regression models](https://github.com/tylerho5/school-projects/blob/main/movie-recommender-system/scripts/RegressionModel.py).
  - [Preprocessing Scaler](https://github.com/scikit-learn/scikit-learn/blob/fa5d7275b/sklearn/preprocessing/_data.py#L710) - Used as a reference for [data preprocessing methodology](https://github.com/tylerho5/school-projects/blob/main/movie-recommender-system/scripts/Scaler.py).
  - [Ranking Metrics](https://github.com/scikit-learn/scikit-learn/blob/46a7c9a5e4fe88dfdfd371bf36477f03498a3390/sklearn/metrics/_ranking.py#L1750) - Consulted for the implementation of ranking metrics, such as AUC and precision-recall.

- **[Gensim](https://github.com/piskvorky/gensim)**:
  - [Word2Vec Model](https://github.com/piskvorky/gensim/blob/develop/gensim/models/word2vec.py) - Used as a reference for understanding and [implementing custom Word2Vec embeddings model](https://github.com/tylerho5/school-projects/blob/main/movie-recommender-system/scripts/Word2Vec.py).

- **[TensorFlow Text](https://github.com/tensorflow/text)**:
  - [Word2Vec Tutorial](https://github.com/tensorflow/text/blob/master/docs/tutorials/word2vec.ipynb) - Consulted for a practical tutorial on [Word2Vec model implementation](https://github.com/tylerho5/school-projects/blob/main/movie-recommender-system/scripts/Word2Vec.py) and usage.

We are grateful to the open-source community for providing these high-quality resources, which have significantly contributed to the inspiration and development of this project.

## Contributors

This project was a collaborative effort between:
- Tyler Ho: Project lead, implementation of custom SVD, custom Word2Vec, custom scaler, custom regression, as well as documentation and codebase organiziation.
- [Quynh Nguyen](https://www.linkedin.com/in/quynhnng/): Contributions in research, testing, technical report, and presentation.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.