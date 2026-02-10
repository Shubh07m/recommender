from flask import Flask, request, jsonify, render_template

import pickle
import pandas as pd
import numpy as np
import sys

app = Flask(__name__)

# Global variables to store model components and dataframes
svd_norm_final = None
user_mean_ratings_final = None
predicted_ratings_df_norm_final = None
course_df = None
ratings_df = None # Needed for get_recommendations function

def load_model_components():
    global svd_norm_final, user_mean_ratings_final, predicted_ratings_df_norm_final, course_df, ratings_df
    print("Loading model components and dataframes...")

    try:
        # Ensure pandas is properly loaded
        if 'pandas' in sys.modules:
            del sys.modules['pandas']
        import pandas as pd

        # Load the trained SVD model object
        with open('svd_norm_final.pkl', 'rb') as f:
            svd_norm_final = pickle.load(f)
        print("svd_norm_final.pkl loaded.")

        # Load the user mean ratings Series
        with open('user_mean_ratings_final.pkl', 'rb') as f:
            user_mean_ratings_final = pickle.load(f)
        print("user_mean_ratings_final.pkl loaded.")

        # Load the predicted ratings DataFrame
        predicted_ratings_df_norm_final = pd.read_csv('predicted_ratings_df_norm_final.csv', index_col='user_id')
        print("predicted_ratings_df_norm_final.csv loaded.")

        # --- Mock data for ratings_df and course_df if not available globally ---
        # In a real scenario, these would be loaded from your data source
        # For demonstration, recreating based on the notebook's known state
        # This ensures the get_recommendations function has its dependencies
        # The `df` variable from the notebook was used as `ratings_df`.
        # The `course_df` variable was also created in the notebook.

        # Recreate course_df based on the notebook's state if not available
        if course_df is None:
            # This mock course_df should match the structure and content from your notebook's `course_df` variable.
            # I will use the variable `course_df` from the kernel state as a reference
            course_data = {
                'course_name': [
                    'Python Basics', 'Data Science Fundamentals', 'Machine Learning 101',
                    'Deep Learning Essentials', 'Web Development Masterclass', 'Mobile App Development with Swift',
                    'Digital Marketing Strategies', 'Financial Modeling and Valuation', 'Graphic Design Fundamentals',
                    'Cybersecurity Basics', 'Cloud Computing with AWS', 'Business Analytics for Managers',
                    'Stock Market and Trading Strategies', 'Artificial Intelligence in Practice', 'Blockchain Fundamentals',
                    'Game Development with Unity', 'Data Visualization with Tableau', 'Creative Writing Workshop',
                    'Project Management Professional (PMP)', 'Photography Masterclass'
                ],
                'difficulty_level': [
                    'Beginner', 'Intermediate', 'Advanced', 'Advanced', 'Intermediate', 'Intermediate',
                    'Beginner', 'Advanced', 'Beginner', 'Beginner', 'Advanced', 'Intermediate',
                    'Advanced', 'Advanced', 'Beginter', 'Intermediate', 'Intermediate', 'Beginner',
                    'Advanced', 'Beginner'
                ],
                'certification_offered': [
                    'Yes', 'Yes', 'No', 'Yes', 'Yes', 'No',
                    'Yes', 'Yes', 'No', 'Yes', 'Yes', 'No',
                    'Yes', 'No', 'Yes', 'No', 'Yes', 'No',
                    'Yes', 'No'
                ],
                'instructor': [
                    'John Doe', 'Jane Smith', 'Alice Johnson', 'Alice Johnson', 'Chris Lee', 'Emily White',
                    'David Brown', 'Michael Green', 'Olivia Taylor', 'Sophia Wilson', 'William Thomas', 'Emma Harris',
                    'Liam Martinez', 'Noah King', 'Isabella Hall', 'James Clark', 'Ava Lewis', 'Ethan Scott',
                    'Mia Adams', 'Alexander Wright'
                ]
            }
            course_df = pd.DataFrame(course_data)
            print("Mock course_df created.")

        if ratings_df is None:
            # This mock ratings_df should match the structure and content from your notebook's `ratings_df` variable.
            # I will use the variable `ratings_df` from the kernel state as a reference.
            ratings_data = {
                'user_id': [1, 1, 1, 2, 2, 3, 3, 4, 4],
                'course_name': [
                    'Python Basics', 'Data Science Fundamentals', 'Machine Learning 101',
                    'Python Basics', 'Deep Learning Essentials', 'Data Science Fundamentals',
                    'Machine Learning 101', 'Python Basics', 'Deep Learning Essentials'
                ],
                'rating': [5, 4, 3, 4, 5, 5, 4, 3, 4]
            }
            ratings_df = pd.DataFrame(ratings_data)
            print("Mock ratings_df created.")

        print("All model components and dataframes loaded successfully.")
    except Exception as e:
        print(f"Error loading model components: {e}")
        sys.exit(1) # Exit if essential components cannot be loaded


def get_recommendations(user_id, num_recommendations=5):
    global ratings_df, course_df, user_mean_ratings_final, predicted_ratings_df_norm_final

    # 1. Check if the user exists in our prediction matrix
    if user_id not in predicted_ratings_df_norm_final.index:
        # Fallback to general popular courses or handle cold start differently
        return pd.DataFrame() # Return an empty DataFrame for simplicity in API

    # 2. Get courses the user has already rated from the original ratings_df
    user_rated_courses = ratings_df[ratings_df['user_id'] == user_id]['course_name'].tolist()

    # 3. Get all unique courses from course_df
    all_courses = course_df['course_name'].unique()

    # 4. Identify courses the user has not yet rated
    unrated_courses = [course for course in all_courses if course not in user_rated_courses]

    if not unrated_courses:
        return pd.DataFrame() # Return an empty DataFrame

    recommendations = []

    # 5. Get the user's mean rating for denormalization
    # Fallback to global mean if user_id not found in user_mean_ratings_final (shouldn't happen for existing users)
    user_mean_rating = user_mean_ratings_final.get(user_id, ratings_df['rating'].mean())

    for course_name in unrated_courses:
        if course_name in predicted_ratings_df_norm_final.columns:
            # Retrieve predicted normalized rating
            predicted_norm_rating = predicted_ratings_df_norm_final.loc[user_id, course_name]

            # Denormalize the rating
            predicted_original_scale_rating = predicted_norm_rating + user_mean_rating

            # Clip to valid rating range (e.g., 1 to 5)
            predicted_final_rating = np.clip(predicted_original_scale_rating, 1, 5)
            recommendations.append((course_name, predicted_final_rating))

    # 6. Sort recommendations by predicted rating in descending order
    sorted_recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)

    # 7. Prepare the final recommendations DataFrame with course details
    final_recommendations_list = []
    for course_name, predicted_rating in sorted_recommendations[:num_recommendations]:
        course_details = course_df[course_df['course_name'] == course_name].iloc[0]
        final_recommendations_list.append({
            'Course Name': course_name,
            'Predicted Rating': round(predicted_rating, 3),
            'Difficulty Level': course_details['difficulty_level'],
            'Certification Offered': course_details['certification_offered'],
            'Instructor': course_details['instructor']
        })

    return pd.DataFrame(final_recommendations_list)


@app.route('/recommend', methods=['GET'])
def recommend():
    user_id_str = request.args.get('user_id')
    num_recommendations_str = request.args.get('num_recommendations', '5')

    if not user_id_str:
        return jsonify({"error": "user_id parameter is required."}), 400

    try:
        user_id = int(user_id_str)
        num_recommendations = int(num_recommendations_str)
    except ValueError:
        return jsonify({"error": "user_id and num_recommendations must be integers."}), 400

    recommendations_df = get_recommendations(user_id, num_recommendations)

    if recommendations_df.empty:
        return jsonify({"message": f"No recommendations found for user {user_id}."}), 200
    else:
        return jsonify(recommendations_df.to_dict(orient='records'))



@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/recommend-ui", methods=["POST"])
def recommend_ui():
    user_id = int(request.form["user_id"])
    num = int(request.form["num_recommendations"])

    df = get_recommendations(user_id, num)
    recommendations = df.to_dict(orient="records") if not df.empty else []

    return render_template("index.html", recommendations=recommendations)


# Load model components when the application starts
with app.app_context():
    load_model_components()

if __name__ == '__main__':
    # Use '0.0.0.0' to make it accessible from outside the container/localhost
    app.run(host='0.0.0.0', port=5000)
