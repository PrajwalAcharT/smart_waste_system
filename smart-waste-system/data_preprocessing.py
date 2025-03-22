import os
import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

logger = logging.getLogger("DataPreprocessing")

def load_and_process(data_path="data", create_dummy=True):
    """
    Load and preprocess data for recommendation systems.
    Parameters:
        data_path: Path to data directory.
        create_dummy: Whether to create dummy data if real data doesn't exist.
    Returns:
        waste_df: DataFrame containing waste information.
        ratings_df: DataFrame containing user ratings.
        waste_features: DataFrame with processed features for content-based filtering.
        user_item_matrix: Matrix of user ratings for collaborative filtering.
    """
    try:
        # Check if data files exist
        waste_file = os.path.join(data_path, "waste_data.csv")
        ratings_file = os.path.join(data_path, "user_ratings.csv")
        if os.path.exists(waste_file) and os.path.exists(ratings_file):
            logger.info("Loading existing data files.")
            waste_df = pd.read_csv(waste_file)
            ratings_df = pd.read_csv(ratings_file)
        elif create_dummy:
            logger.info("Creating dummy data for initial setup.")
            waste_df, ratings_df = create_dummy_data()
            # Save dummy data
            os.makedirs(data_path, exist_ok=True)
            waste_df.to_csv(waste_file, index=False)
            ratings_df.to_csv(ratings_file, index=False)
        else:
            logger.error("Data files not found and dummy creation disabled.")
            return None, None, None, None

        # Process waste features
        waste_features = extract_waste_features(waste_df)

        # Create user-item matrix for collaborative filtering
        user_item_matrix = create_user_item_matrix(ratings_df)

        return waste_df, ratings_df, waste_features, user_item_matrix
    except Exception as e:
        logger.error(f"Error in data preprocessing: {str(e)}")
        return None, None, None, None


def extract_waste_features(waste_df):
    """Extract and normalize features for content-based recommendation."""
    # Select feature columns
    feature_cols = [col for col in waste_df.columns if col.startswith('feature_')]
    if not feature_cols:
        # If no feature columns exist, create default features
        logger.warning("No feature columns found, creating default features.")
        waste_df['feature_cellulose'] = waste_df['material_category'].apply(
            lambda x: 0.8 if x == 'Cellulosic' else 0.3
        )
        waste_df['feature_nitrogen'] = waste_df['material_category'].apply(
            lambda x: 0.7 if x == 'Protein-rich' else 0.2
        )
        waste_df['feature_lignin'] = waste_df['material_category'].apply(
            lambda x: 0.9 if x == 'Woody' else 0.4
        )
        feature_cols = ['feature_cellulose', 'feature_nitrogen', 'feature_lignin']

    # Extract features and normalize
    features = waste_df[feature_cols].copy()
    features.fillna(features.mean(), inplace=True)  # Handle missing values
    scaler = MinMaxScaler()
    scaled_features = pd.DataFrame(
        scaler.fit_transform(features),
        index=features.index,
        columns=features.columns
    )
    return scaled_features


def create_user_item_matrix(ratings_df):
    """Create user-item matrix for collaborative filtering."""
    # Create unique identifiers for items (waste_id, application pairs)
    ratings_df['item_id'] = ratings_df.apply(
        lambda row: f"{row['waste_id']}_{row['application']}",
        axis=1
    )
    # Create pivot table
    user_item = ratings_df.pivot(
        index='user_id',
        columns='item_id',
        values='rating'
    )
    return user_item


def create_dummy_data():
    """Create dummy data for initial setup."""
    # Create waste data
    waste_types = [
        'wheat', 'rice', 'corn', 'sugarcane', 'cotton',
        'soybean', 'coconut', 'coffee', 'banana', 'paddy'
    ]
    material_categories = ['Cellulosic', 'Protein-rich', 'Fibrous', 'Woody', 'Mixed']
    processing_difficulties = ['Low', 'Medium', 'High']
    waste_data = []
    for i, waste in enumerate(waste_types):
        waste_data.append({
            'waste_id': f"w{i+1}",
            'waste_name': waste,
            'material_category': np.random.choice(material_categories),
            'processing_difficulty': np.random.choice(processing_difficulties),
            'feature_cellulose': np.random.uniform(0.2, 0.9),
            'feature_nitrogen': np.random.uniform(0.1, 0.8),
            'feature_lignin': np.random.uniform(0.3, 0.9),
            'feature_moisture': np.random.uniform(0.2, 0.7)
        })
    waste_df = pd.DataFrame(waste_data)

    # Add application suitability
    applications = ['Compost', 'Biogas', 'Animal Feed', 'Mulch', 'Craft', 'Biofuel']
    for app in applications:
        waste_df[f'app_{app}'] = np.random.uniform(0.2, 0.9, size=len(waste_df))

    # Create dummy ratings
    user_ids = [f"user_{i}" for i in range(1, 21)]
    ratings_data = []
    for user_id in user_ids:
        num_ratings = np.random.randint(3, 6)
        for _ in range(num_ratings):
            waste_id = waste_df.sample(1)['waste_id'].values[0]
            application = np.random.choice(applications)
            rating = np.random.randint(1, 6)
            ratings_data.append({
                'user_id': user_id,
                'waste_id': waste_id,
                'application': application,
                'rating': rating,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
    ratings_df = pd.DataFrame(ratings_data)
    return waste_df, ratings_df


def update_ratings_from_log(ratings_df, log_file='logs/feedback.log'):
    """Update ratings dataframe from feedback log file."""
    try:
        if not os.path.exists(log_file):
            logger.warning(f"Feedback log file {log_file} not found.")
            return ratings_df

        # Read log file
        log_data = []
        with open(log_file, 'r') as f:
            for line in f:
                try:
                    timestamp, user_id, waste_id, application, rating = line.strip().split(',')
                    log_data.append({
                        'user_id': user_id,
                        'waste_id': waste_id,
                        'application': application,
                        'rating': float(rating),
                        'timestamp': timestamp
                    })
                except ValueError:
                    continue

        if not log_data:
            return ratings_df

        # Convert to DataFrame
        log_df = pd.DataFrame(log_data)
        # Combine with existing ratings
        updated_df = pd.concat([ratings_df, log_df], ignore_index=True)
        # Remove duplicates, keeping the latest rating
        updated_df['timestamp'] = pd.to_datetime(updated_df['timestamp'], errors='coerce')
        updated_df.sort_values('timestamp', inplace=True)
        updated_df = updated_df.drop_duplicates(
            subset=['user_id', 'waste_id', 'application'],
            keep='last'
        )
        return updated_df
    except Exception as e:
        logger.error(f"Error updating ratings from log: {str(e)}")
        return ratings_df


def save_data(waste_df, ratings_df, data_path="data"):
    """Save processed data to files."""
    try:
        os.makedirs(data_path, exist_ok=True)
        waste_file = os.path.join(data_path, "waste_data.csv")
        ratings_file = os.path.join(data_path, "user_ratings.csv")
        waste_df.to_csv(waste_file, index=False)
        ratings_df.to_csv(ratings_file, index=False)
        logger.info(f"Data saved to {data_path}.")
        return True
    except Exception as e:
        logger.error(f"Error saving data: {str(e)}")
        return False