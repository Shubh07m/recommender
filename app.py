from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
import os

app = Flask(__name__)

# Global variables
svd_model = None
user_mean_ratings = None
ratings_df = None
user_item_matrix = None
global_mean = 0

def build_recommendation_engine():
    """
    Builds the recommendation engine from the CSV data.
    Uses corrected normalization to prevent 0.00 scores.
    """
    global svd_model, user_mean_ratings, ratings_df, user_item_matrix, global_mean
    
    try:
        print("Initializing Recommendation Engine...")
        ratings_df = pd.read_csv('processed_online_course_data.csv')
        
        # 1. Create the User-Item Matrix
        user_item_matrix = ratings_df.pivot_table(
            index='user_id', 
            columns='course_name', 
            values='rating'
        ).fillna(0)
        
        # 2. CORRECTED: Calculate mean of ONLY actual ratings (ignore zeros)
        # This prevents the '0.00' score issue
        user_mean_ratings = user_item_matrix.replace(0, np.nan).mean(axis=1)
        global_mean = ratings_df['rating'].mean()
        user_mean_ratings = user_mean_ratings.fillna(global_mean)
        
        # 3. Normalize the Matrix
        # Subtract user mean from their ratings; missing values remain 0 (average)
        matrix_norm = user_item_matrix.replace(0, np.nan).sub(user_mean_ratings, axis=0).fillna(0)
        
        # 4. Train SVD
        print("Training Model...")
        # Using 15 components for a balance of speed and accuracy
        n_comp = min(15, user_item_matrix.shape[1] - 1)
        svd_model = TruncatedSVD(n_components=n_comp, random_state=42)
        svd_model.fit(matrix_norm)
        
        print("Engine Ready!")
        
    except Exception as e:
        print(f"Startup Error: {e}")

def get_recommendations(user_id, num_recs=5):
    """Generates accurate predicted scores for courses."""
    if user_id not in user_item_matrix.index:
        return pd.DataFrame()

    # Get user info
    u_mean = user_mean_ratings[user_id]
    
    # Prepare normalized row for prediction
    # We take the actual ratings the user gave, subtract their mean, and fill rest with 0
    user_row = user_item_matrix.loc[[user_id]]
    user_row_norm = user_row.replace(0, np.nan).sub(u_mean).fillna(0)
    
    # SVD Prediction
    latent_vec = svd_model.transform(user_row_norm)
    reconstructed_norm = svd_model.inverse_transform(latent_vec)
    
    # De-normalize (Add the mean back to get the 1-5 star score)
    final_scores = reconstructed_norm[0] + u_mean
    
    # Clean up scores (ensure they stay within 1-5 range)
    final_scores = np.clip(final_scores, 1.0, 5.0)
    
    results = pd.DataFrame({
        'course_name': user_item_matrix.columns,
        'predicted_rating': final_scores
    })
    
    # Don't recommend courses they already took
    taken = ratings_df[ratings_df['user_id'] == user_id]['course_name'].unique()
    filtered = results[~results['course_name'].isin(taken)]
    
    return filtered.sort_values(by='predicted_rating', ascending=False).head(num_recs)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend-ui', methods=['POST'])
def recommend_ui():
    try:
        u_id_input = request.form.get('user_id')
        if not u_id_input:
            return render_template('index.html', error="Please enter a User ID.")
            
        u_id = int(u_id_str := u_id_input)
        num = int(request.form.get('num_recommendations', 5))
        
        recs_df = get_recommendations(u_id, num)
        
        if recs_df.empty:
            return render_template('index.html', error=f"User ID {u_id} not found.")
            
        return render_template('index.html', 
                               recommendations=recs_df.to_dict(orient='records'),
                               user_id=u_id)
    except ValueError:
        return render_template('index.html', error="User ID must be a number.")
    except Exception as e:
        return render_template('index.html', error=f"System Error: {str(e)}")

if __name__ == '__main__':
    build_recommendation_engine()
    app.run(host='0.0.0.0', port=5000, debug=False)