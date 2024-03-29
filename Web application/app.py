from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import tensorflow as tf


# Initialize Flask application
app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('model/model.h5')

# Load dataset
dataset = pd.read_csv('dataset.csv')
books_dataset = pd.read_csv('books_dataset.csv')

# Function to recommend books for a given user_id
def recommend_books_id(user_id):
    book_data = np.array(list(set(dataset.book_id)))
    user = np.array([user_id for i in range(len(book_data))])
    predictions = model.predict([user, book_data])

    unique_book_ids = dataset['book_id'].unique()
    unique_book_ids.sort()

    predictions = np.array([a[0] for a in predictions])
    predictions_sorted = (-predictions).argsort()[:20]
    recommended_book_ids = np.array([unique_book_ids[a] for a in predictions_sorted])

    return recommended_book_ids

def recommend_books_data(books_id, user_id):
    users_books = dataset[dataset['user_id'] == user_id]
    users_fav = users_books.sort_values(by='rating', ascending=False)
    users_fav_dataset = books_dataset[books_dataset['book_id'].isin(users_fav['book_id'])]
    users_fav_dataset = users_fav_dataset.head(10).to_dict(orient='records')

    recommended_book_dataset = books_dataset[books_dataset['book_id'].isin(books_id)]
    recommended_book_dataset = recommended_book_dataset.to_dict(orient='records')
    return (users_fav_dataset, recommended_book_dataset)


# Define route for the index page
@app.route('/')
def index():
    return render_template('index.html')

# Define route for recommendation page
@app.route('/recommend', methods=['POST'])
def recommend():
    if request.method == 'POST':
        user_id = int(request.form['user_id'])
        books_id = recommend_books_id(user_id)
        (users_fav_dataset, recommended_book_dataset) = recommend_books_data(books_id, user_id)
        return render_template('recommend.html', 
                               user_id=user_id, 
                               users_fav_dataset=users_fav_dataset,
                               recommended_book_dataset=recommended_book_dataset)

if __name__ == '__main__':
    app.run(debug=True)