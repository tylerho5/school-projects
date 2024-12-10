### Tyler Ho, Quynh Nguyen
### CS439
### Final Project
### Fall 2024

import pandas as pd
import numpy as np
import time
from joblib import Parallel, delayed
import re
import os

# import custom classes
import RegressionModel as rm
import Word2Vec as w2v
import SVD as sv
import Scaler as sc



start_time = time.time()
print(f"Program started at: {time.ctime(start_time)}")

current_dir = os.path.dirname(os.path.abspath(__file__))

ratings_path = os.path.join(current_dir, '/raw-datasets/ratings.csv')
movies_path = os.path.join(current_dir, '/raw-datasets/movies.csv')
tags_path = os.path.join(current_dir, '/raw-datasets/tags.csv')

ratings = pd.read_csv(ratings_path)
movies = pd.read_csv(movies_path)
tags = pd.read_csv(tags_path)

# convert ids to ints to prevent typecasting
ratings['userId'] = ratings['userId'].astype(int)
ratings['movieId'] = ratings['movieId'].astype(int)
tags['userId'] = tags['userId'].astype(int)
tags['movieId'] = tags['movieId'].astype(int)

ratings.drop('timestamp', axis = 1, inplace = True)
tags.drop('timestamp', axis = 1, inplace = True)
# function to split each user's ratings into two subsets
def custom_train_test_split_per_user(ratings):

    train_list = []
    test_list = []

    # group ratings by user
    ratingsByUser= ratings.groupby('userId')

    for userID, user in ratingsByUser:

        # puts all the raters of users with less than 14 users into training data
        # due to not enough ratings for the test set to have at least 10 ratings
        if len(user) < 50:
            train_list.append(user)
        else:
            user_seed = 69 + userID
            # shuffle ratings for each user
            shuffled_group = user.sample(frac = 1, random_state = user_seed).reset_index(drop = True)
            n_test = int(len(user) * 0.2)  # Number of test samples
            test = shuffled_group.iloc[:n_test]    # Select the first n_test samples as test
            train = shuffled_group.iloc[n_test:]  # Remaining samples as train
            train_list.append(train)
            test_list.append(test)

    train_ratings = pd.concat(train_list).reset_index(drop=True)
    test_ratings = pd.concat(test_list).reset_index(drop=True)
    return train_ratings, test_ratings

trainRatings, testRatings = custom_train_test_split_per_user(ratings)

# reshape df to have users as rows and movies as cols
# missing ratings are set to 0
userItemMatrix = trainRatings.pivot(index = 'userId', columns = 'movieId', values = 'rating').fillna(0)

# initiate svd model
svd1 = sv.SVD()
# perform svd on df
user_features = svd1.fit_transform(userItemMatrix)
# (V^T)^T = V
item_features = svd1.components.T 

# dict to map user id to index in df
userToIndex = {id: index for index, id in enumerate(userItemMatrix.index)}

# attach tags to ratings
ratingsTagged = tags.merge(trainRatings, on=['userId', 'movieId'], how='inner')

# standardize tag format
ratingsTagged['tag'] = ratingsTagged['tag'].fillna("").str.lower().str.strip()

ratingsTagged['tag_weight'] = ratingsTagged['rating']

# group by user and tag to get average tag weight for each user's tag
userntagPref = (ratingsTagged.groupby(['userId', 'tag'])['tag_weight'].mean().reset_index())

### handle genres

# split main genres string into individual genres
movies['genres'] = movies['genres'].apply(lambda x: x.split('|') if isinstance(x, str) else [])

# maps genres to movie ids
movie_id_to_genres = dict(zip(movies['movieId'], [[genre.lower() for genre in genres] for genres in movies['genres']]))
    
# get set of unique genres
unique_genres = set(genre for genres in movie_id_to_genres.values() for genre in genres)

# group by user and genre and calculate average genre weight
# sets genre weight a tag weight if tag matches genre
tags_with_genres = ratingsTagged[ratingsTagged['tag'].isin(unique_genres)].copy()
user_genre_preferences = (tags_with_genres.groupby(['userId', 'tag'])['tag_weight'].mean().reset_index().rename(columns={'tag': 'genre'}))

# pivot to set up user to genre matrix properly
user_genre_matrix = user_genre_preferences.pivot(index='userId', columns='genre', values='tag_weight').fillna(0)

# reindex to make sure all genres are included
user_genre_matrix = user_genre_matrix.reindex(columns=unique_genres, fill_value=0)
    
# map user id to index
userToGenre = {id: index for index, id in enumerate(user_genre_matrix.index)}

# users without genre preference
missing_users = set(userItemMatrix.index) - set(user_genre_matrix.index)
    
if not user_genre_matrix.columns.empty:
    if missing_users:
        # df for users with no preferences
        zero_preferences = pd.DataFrame(
            0,
            index=list(missing_users),
            columns=user_genre_matrix.columns
        )

        # add zero preference matrix to the main user to genre matrix
        user_genre_matrix = pd.concat([user_genre_matrix, zero_preferences])
    
    # reindex to match the user to genre matrix to the user to item matrix order
    user_genre_matrix = user_genre_matrix.reindex(userItemMatrix.index).fillna(0.0)

### combine tags and genres into embeddings

# combine tags and genres to be trained in Word2Vec
corpus = userntagPref['tag'].tolist() + list(unique_genres)

# initialize Word2Vec model
word2vec = w2v.Word2Vec()

# give tags and genres to model
word2vec.train([corpus]) 

# get average embedding from user's tags and watched genres
def getUserEmbedding(user):

    # get tags for user
    user_tags = userntagPref[userntagPref['userId'] == user]['tag'].tolist()

    # get the genres user likes
    user_genres = user_genre_matrix.loc[user][user_genre_matrix.loc[user] > 0].index.tolist()
    tokens = user_tags + user_genres

    # if user does have preferred genres and tags
    if tokens:

        # average the embeddings of all tokens
        embeddings = [word2vec.getEmbedding(token) for token in tokens]
        return np.mean(embeddings, axis=0)
    
    # if user has no prefered genre or tags
    else:
        return np.ones(word2vec.vecSize) 

# get embeddings for every user and puts in matrix
user_embedding_matrix = np.array([getUserEmbedding(user) for user in userItemMatrix.index])

# normalizes the user and genre/tag embeddings
user_scaler = sc.Scaler()
user_embedding_matrix_scaled = user_scaler.fit_transform(user_embedding_matrix)

# gets average embedding from movie's tags and genres
def create_item_embedding(movie_id):

    # gets movie's tags
    movie_tags = ratingsTagged[ratingsTagged['movieId'] == movie_id]['tag'].tolist()

    # get movie's genres
    movie_genres = movie_id_to_genres[movie_id]
    tokens = movie_tags + movie_genres

    # if movie does have genre and tags
    if tokens:

        # average the embeddings of all tokens
        embeddings = [word2vec.getEmbedding(token) for token in tokens]
        return np.mean(embeddings, axis=0)
    
    # if movie has no genre or tags
    else:
        return np.ones(word2vec.vecSize) 

# create embeddings for all movies
item_embedding_matrix = np.array([create_item_embedding(movie_id) for movie_id in userItemMatrix.columns])

# normalize movie embeddings
item_scaler = sc.Scaler()
item_embedding_matrix_scaled = item_scaler.fit_transform(item_embedding_matrix)

# combine svd user features with normalized user embeddings
combined_user_features = np.hstack([user_features, user_embedding_matrix_scaled])

# combine svd movie features with normalized movie embeddings
combined_item_features = np.hstack([item_features, item_embedding_matrix_scaled])

# list to hold feature vectors and ratings
features = []
ratings = []

# build feature vector for each row in training set 
for _, row in trainRatings.iterrows():
    user = row['userId']
    movie_id = row['movieId']

    # get user and movie indexes
    userIndex = userToIndex[user]
    movieIndex = userItemMatrix.columns.get_loc(movie_id)

    # get correspondng feature vectors
    user_feat = combined_user_features[userIndex]
    item_feat = combined_item_features[movieIndex]

    # combine user and item features 
    features.append(np.concatenate([user_feat, item_feat]))
    
    # add ratings to vector
    ratings.append(row['rating'])

# convert to numpy array
features = np.array(features)
ratings = np.array(ratings)

# initialize regression model and fit it to user-movie matrix
regression_model = rm.RegressionModel(alpha=5)
regression_model.fit(features, ratings)

# save coefficients for rating prediction
coefficients = regression_model.coef

### apply regression model to test set 

# predict user ratings on test set movies
def predRatingRegression(user, movie_id):

    # if user hasn't watched movie
    if user not in userToIndex or movie_id not in userItemMatrix.columns:
        return np.nan, {}
    
    # get user and movie indexes
    userIndex = userToIndex[user]
    movieIndex = userItemMatrix.columns.get_loc(movie_id)

    # get users and movie features 
    user_feat = combined_user_features[userIndex]
    item_feat = combined_item_features[movieIndex]

    # consider both feature vectors as a single vector
    feature = np.concatenate([user_feat, item_feat]).reshape(1, -1)

    # apply regression model to rating
    predictedRating = regression_model.predict(feature)[0]
    
    # determine how much features contribute to rating prediction
    if coefficients is not None:
        feature_vector = feature.flatten()
        contributions = feature_vector * coefficients
        totalCont = np.sum(contributions)
        
        # determine where the user features end within matrix
        split = len(combined_user_features[0])

        # split along determined line
        userFeatCont = contributions[:split]
        itemFeatCont = contributions[split:]

        # user features/contributions as first half of matrix and tags + genres features/contributions as second half
        half = split // 2
        simUserCont = np.sum(userFeatCont[:half]) + np.sum(itemFeatCont[:half])
        itemFeatCont = np.sum(userFeatCont[half:]) + np.sum(itemFeatCont[half:])
        
        # calculate percentage of contributions by similar users and tags + genres
        if totalCont != 0:
            similarUserPercent = (simUserCont / totalCont) * 100
            itemFeatPercent = (itemFeatCont / totalCont) * 100
        else:
            similarUserPercent = 0
            itemFeatPercent = 0
        
        contributions_dict = {
            'similar_user_ratings': similarUserPercent,
            'tags_and_genres': itemFeatPercent
        }
    else:
        contributions_dict = {}
    
    return predictedRating, contributions_dict

# apply predictions to the test set with contributions
testRatings['predicted_rating'], testRatings['contributions'] = zip(*testRatings.apply(
    lambda row: predRatingRegression(row['userId'], row['movieId']), axis=1))

# remove blank predictions
testRatings_clean = testRatings.dropna(subset=['predicted_rating'])

### get contributions for each user 

# dict to store user contributions
user_contributions = {}

# add up all of user contributions over test set 
for _, row in testRatings_clean.iterrows():

    #sets user and contribution to row in test set 
    user = row['userId']
    contrib = row['contributions']

    # skip if no user contributions for a movie
    if not contrib:
        continue 

    # adds all contributions 
    if user not in user_contributions:
        user_contributions[user] = {'similar_user_ratings': 0, 'tags_and_genres': 0}
    user_contributions[user]['similar_user_ratings'] += contrib['similar_user_ratings']
    user_contributions[user]['tags_and_genres'] += contrib['tags_and_genres']

# maps user to to movies they've rated in test set 
userTestMovies = testRatings_clean.groupby('userId')['movieId'].apply(set).to_dict()

# averages contributions per user 
for user in user_contributions:
    count = len(userTestMovies.get(user, []))
    if count > 0:
        user_contributions[user]['similar_user_ratings'] /= count
        user_contributions[user]['tags_and_genres'] /= count

### recommend movies to user 

# generates recommendations from all movies for every user 
def getRecommendations(user):
    
    # gets user index in feature matrix
    userIndex = userToIndex[user]
    user_feat = combined_user_features[userIndex].reshape(1, -1)

    allMoviesInTraining = set(userItemMatrix.columns)

    userRatedMovies = set(trainRatings[trainRatings['userId'] == user]['movieId'])

    itemsToRec = allMoviesInTraining - userRatedMovies

    # case for users without any ratings in test set 
    if not itemsToRec:
        return []  

    # get indexes of test items
    itemIndices = [userItemMatrix.columns.get_loc(movie_id) for movie_id in itemsToRec if movie_id in userItemMatrix.columns]

    # get features for items
    itemFeats = combined_item_features[itemIndices]

    # repeat user features for all test ratings
    userFeatMatrix = np.repeat(user_feat, len(itemFeats), axis=0)

    #combine user and rating features
    combinedFeats = np.hstack([userFeatMatrix, itemFeats])

    # apply regression to predict ratings for test set
    scores = regression_model.predict(combinedFeats)

    # map ratings to movie ids for ranking
    recommendedMovieIDS = [userItemMatrix.columns[i] for i in itemIndices]
    scores_series = pd.Series(scores, index =recommendedMovieIDS)

    # sort predicted ratings in descending order and get top 10
    top_n = scores_series.sort_values(ascending=False).head(10).index.tolist()

    return top_n

# get recommendationos for all users in test set 
test_users_unique = testRatings['userId'].unique()
# parallelize the job to reduce runtime
recommendations = Parallel(n_jobs=-1)(delayed(getRecommendations)(user) for user in test_users_unique)
recsDict = dict(zip(test_users_unique, recommendations))

### format results 

# reformats movie titles
def reformatTitle(title):
    
    articles = ['the', 'a', 'an']
    
    # separates parts of title by main title, beginning the, a, or an, and release year
    match = re.match(r'^(.*),\s+(The|A|An)\s*\((\d{4})\)$', title, re.IGNORECASE)

    if match:
        main_title = match.group(1)
        article = match.group(2)
        year = match.group(3)
        article = article.capitalize()
        main_title = main_title.title()

        # returns rearranged title
        return f"{article} {main_title} ({year})"

    else:
        return title

# maps movie titles to id
movie_id_to_title = dict(zip(movies['movieId'], movies['title']))

# attaches explanation for recommendation to each movie
def getExplanations(user, recommended_movies):
    explanations = []
    # gets the users tags and preferred genres
    user_tags = userntagPref[userntagPref['userId'] == user]['tag'].tolist()
    user_genres = user_genre_matrix.loc[user][user_genre_matrix.loc[user] > 0].index.tolist()
    
    # attaches explanation for top 3 recommended movies
    for movie_id in recommended_movies[:3]:

        # converts id to title
        movie_title = movie_id_to_title.get(movie_id, f"Movie ID {movie_id}")
        
        #cleans title formatting
        movie_title_clean = reformatTitle(movie_title)

        movie_genres = movie_id_to_genres.get(movie_id, list())
        movie_tags = ratingsTagged[ratingsTagged['movieId'] == movie_id]['tag'].tolist()
        
        # check how many user genres and tags line up with tags and genres for movie
        common_genres = set(user_genres).intersection(set(movie_genres))
        common_tags = set(user_tags).intersection(set(movie_tags))
        
        explanation_parts = []
        if common_genres:
            genres_str = ", ".join(common_genres)
            explanation_parts.append(f"user likes {genres_str} genres")
        if common_tags:
            tags_str = ", ".join(common_tags)
            explanation_parts.append(f"user likes movies tagged with {tags_str}")
        if not explanation_parts:
            explanation_parts.append("similar users liked it")
        
        explanation = f"{movie_title_clean} recommended because {' and '.join(explanation_parts)}."
        explanations.append(explanation)
    
    return explanations

# get explanations for all users
explanationDict = {}
for user, recommendations in recsDict.items():
    explanations = getExplanations(user, recommendations)
    explanationDict[user] = explanations

# format movie id recommendations
def format_recommendations_ids(recommendationDict):
    
    lines = []
    for user, movie_list in recommendationDict.items():
        if isinstance(user, (np.integer, np.int64)):
            user = int(user)

        movie_str_list = [str(movie_id) for movie_id in movie_list]

        movies_str = ", ".join(movie_str_list)

        # formats line for clear reading
        line = f"{user}\t{movies_str}"
        lines.append(line)
    # combine all user lines
    formatted_string = "\n".join(lines)
    return formatted_string

# format movie title recommendations
def format_recommendations_titles(recommendationsDict, movie_mapping):

    lines = []
    for user, movie_list in recommendationsDict.items():
        if isinstance(user, (np.integer, np.int64)):
            user = int(user)
        # convert movie ids to titles
        movie_titles = [movie_mapping.get(movie_id) for movie_id in movie_list]

        # fixes movie title formatting
        movie_titles_clean = [reformatTitle(title) for title in movie_titles]
        
        movies_str = ", ".join(movie_titles_clean)

        # formats line for clear reading
        line = f"{user}\t{movies_str}"
        lines.append(line)
    
    # combine all user lines
    formatted_string = "\n".join(lines)
    return formatted_string

# format user contribution weighting
def format_user_contributions(contributionDict):

    lines = []
    for user, contrib in contributionDict.items():
        similar_pct = contrib.get('similar_user_ratings', 0.0)
        tags_pct = contrib.get('tags_and_genres', 0.0)
        line = f"{user}\tSimilar User Ratings: {similar_pct:.2f}%, Tags & Genres: {tags_pct:.2f}%"
        lines.append(line)
    formatted_string = "\n".join(lines)
    return formatted_string

# format recommendation explanations to string for writing to file 
def formatRecExplan(explanationDict):

    lines = []
    for user, explanations in explanationDict.items():
        if isinstance(user, (np.integer, np.int64)):
            user = int(user)

        # join explanations single string
        explanations_str = " | ".join(explanations)

        # formats user line
        line = f"{user}\t{explanations_str}"
        lines.append(line)

    # combine all user lines
    formatted_string = "\n".join(lines)
    return formatted_string

# format recommendations to write to file
fmtRecswID = format_recommendations_ids(recsDict)
fmtRecswTitles = format_recommendations_titles(recsDict, movie_id_to_title)
fmtUserWeighting = format_user_contributions(user_contributions)
fmtRecExplan = formatRecExplan(explanationDict)

### write results to file

# recommendations with movie IDs
with open('recommendations_ids.txt', 'w') as output:
    output.write(fmtRecswID)
print("Movie ID recommendations have been written to 'recommendations_ids.txt'.")

# recommendations with movite titles
with open('recommendations_titles.txt', 'w') as output:
    output.write(fmtRecswTitles)
print("Movie Title recommendations have been written to 'recommendations_titles.txt'.")

# user contribution weightings
with open('recommendations_user_contributions.txt', 'w') as output:
    output.write(fmtUserWeighting)
print("User preference contributions have been written to 'user_contributions.txt'.")

# recommendation explanations
with open('recommendations_explanations.txt', 'w') as output:
    output.write(fmtRecExplan)
print("Recommendation explanations have been written to 'recommendations_explanations.txt'.")

### evaluate results

# calculate mae
def get_mae(actual, predicted):
    return np.mean(np.abs(actual - predicted))

# calculate RMSE
def get_rmse(actual, predicted):
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    return rmse

# calculate precision
def get_prec(recommended, test_set):

    # covers edge case where divide by zero error occurs
    if not recommended:
        return 0.0
    relevant = recommended & test_set
    precision = len(relevant) / 10
    return precision

# calculate recall
def get_recall(recommended, test_set):

    # covers edge case where divide by zero error occurs
    if not test_set:
        return 0.0
    relevant = recommended & test_set
    recall = len(relevant) / len(test_set)
    return recall

# calculate f-measure
def get_f_m(precision, recall):
    
    # covers edge case where divide by zero error occurs
    if precision + recall == 0:
        return 0
    f1 = 2 * precision * recall / (precision + recall)
    return f1

def get_NDCG(recommended, test_set):
    
    # get top 10 recommended
    recommended = recommended[:10]
    
    # set up array where only relevant movies contribute to DCG
    relevance = np.array([1 if movie_id in test_set else 0 for movie_id in recommended])
    
    #calculate DCG
    discounts = 1 / np.log2(np.arange(2, 2 + len(recommended)))
    dcg = np.sum(relevance * discounts)
    
    # calculate IDCG
    ideal_relevance = np.ones(min(len(test_set), 10))
    ideal_discounts = 1 / np.log2(np.arange(2, 2 + len(ideal_relevance)))
    idcg = np.sum(ideal_relevance * ideal_discounts)
    
    # covers edge case where divide by zero error occurs
    if idcg == 0:
        return 0.0
    
    # Compute NDCG
    ndcg = dcg / idcg
    return ndcg

prec_list = []
recall_list = []
f_m_list = []
ndcg_list = []

# calcuates precision, recall, f-measure, and NDCG for each user
for user, recommendations in recsDict.items():
    test_set = userTestMovies.get(user, set())
    recommended_set = set(recommendations)
    
    prec = get_prec(recommended_set, test_set)
    prec_list.append(prec)
    
    recall = get_recall(recommended_set, test_set)
    recall_list.append(recall)
    
    f_m = get_f_m(prec, recall)
    f_m_list.append(f_m)
    
    ndcg = get_NDCG(recommendations, test_set)
    ndcg_list.append(ndcg)

# recommendation metrics
# calcuate precision, recall, f-measure, NDCG
average_prec = np.mean(prec_list)
average_recall = np.mean(recall_list)
average_f_m = np.mean(f_m_list)
average_NDCG = np.mean(ndcg_list)

# prediction accuracy metrics
# calculate MAE and RMSE 
mae1 = get_mae(testRatings_clean['rating'].values, testRatings_clean['predicted_rating'].values)
rmse1 = get_rmse(testRatings_clean['rating'].values, testRatings_clean['predicted_rating'].values)

### calculate baseline prediction accuracy 

global_mean = trainRatings['rating'].mean()

baseline_predictions = np.full_like(testRatings_clean['rating'].values, global_mean)

baseline_mae = get_mae(testRatings_clean['rating'].values, baseline_predictions)
baseline_rmse = get_rmse(testRatings_clean['rating'].values, baseline_predictions)

# print baseline evaluation metrics
print(f"Baseline Mean Absolute Error (MAE): {baseline_mae:.4f}")
print(f"Baseline Root Mean Squared Error (RMSE): {baseline_rmse:.4f}\n")

# printevaluation metrics
print(f"Mean Absolute Error (MAE): {mae1}")
print(f"Root Mean Squared Error (RMSE): {rmse1}\n")
print(f"Average Precision for top 10: {average_prec:.4f}")
print(f"Average Recall for top 10: {average_recall:.4f}")
print(f"Average F-measure for top 10: {average_f_m:.4f}")
print(f"Average NDCG for top 10: {average_NDCG:.4f}\n")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Program finished at: {time.ctime(end_time)}")
print(f"Total execution time: {elapsed_time / 60:.2f} minutes.")