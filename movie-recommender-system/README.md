# Movie Recommender System

## Description
This project is a movie recommendation system that predicts what users will rate movies they haven't watched yet within the train set and reports genre and tag weighting to explain how recommendations are made.

## Features
- Hybrid filter approach to take advantage of user metadata
- Collaborative filtering through SVD
- Content-based filtering through Word2Vec to analyze genres and tags
- Recommendation explainability and transparency

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/movie-recommender-system.git
    cd movie-recommender-system

2. Create a virtual environment and install dependencies:
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

## Usage
1. Place your datasets in the `datasets/` folder.
2. Run the main script:
    ```bash
    python recSys_final.py
3. Outputs will be saved in the `output/` directory

## Folder Structure
├── output/             # Contains recommendation results
├── raw-datasets/       # Contains raw datasets
├── scripts/            # Python scripts for recommender system
├── requirements.txt    # List of dependencies
└── README.md           # Project documentation

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.