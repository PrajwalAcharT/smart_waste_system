import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
import logging

logger = logging.getLogger("CollaborativeRecommender")


class CollaborativeRecommender:
    """Recommendation system based on collaborative filtering of user preferences."""

    def __init__(self, user_item_matrix, ratings_df, waste_df, min_ratings=3):
        self.user_item_matrix = user_item_matrix
        self.ratings_df = ratings_df
        self.waste_df = waste_df
        self.min_ratings = min_ratings
        self.user_to_idx = {user_id: idx for idx, user_id in enumerate(user_item_matrix.index)}
        self.idx_to_user = {idx: user_id for user_id, idx in self.user_to_idx.items()}
        self.items_df = self._prepare_items_df()
        self.initialize_model()

    def _prepare_items_df(self):
        """Prepare items dataframe from ratings data."""
        items = self.ratings_df[['waste_id', 'application']].drop_duplicates()
        items['item_id'] = range(len(items))
        items.set_index('item_id', inplace=True)
        self.item_to_idx = {}
        for item_id, row in items.iterrows():
            key = (row['waste_id'], row['application'])
            self.item_to_idx[key] = item_id
        self.idx_to_item = {idx: key for key, idx in self.item_to_idx.items()}
        return items

    def initialize_model(self, factors=20):
        """Initialize SVD model if enough data is available."""
        if self.user_item_matrix.shape[0] < self.min_ratings or self.user_item_matrix.shape[1] < self.min_ratings:
            logger.warning("Not enough data for collaborative filtering.")
            self.model_ready = False
            return
        try:
            matrix = self.user_item_matrix.fillna(0).values
            n_factors = min(factors, min(matrix.shape) - 1)
            U, sigma, Vt = svds(matrix, k=n_factors)
            self.U = U
            self.sigma = np.diag(sigma)
            self.Vt = Vt
            self.model_ready = True
            logger.info(f"SVD model initialized with {n_factors} factors.")
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            self.model_ready = False

    def recommend_for_user(self, user_id, top_n=5, include_rated=False):
        """Generate recommendations for a specific user."""
        try:
            if user_id not in self.user_to_idx:
                logger.warning(f"User {user_id} not found, using default recommendations.")
                return self.get_popular_recommendations(top_n)
            user_idx = self.user_to_idx[user_id]
            actual_ratings = self.user_item_matrix.iloc[user_idx].dropna()
            rated_indices = [self.item_to_idx.get((self.items_df.loc[int(idx)]['waste_id'],
                                                   self.items_df.loc[int(idx)]['application']))
                             for idx in actual_ratings.index]
            if self.model_ready:
                predictions = self.predict_ratings(user_idx)
                pred_df = pd.DataFrame({'item_idx': range(len(predictions)), 'prediction': predictions})
                if not include_rated:
                    pred_df = pred_df[~pred_df['item_idx'].isin(rated_indices)]
                pred_df = pred_df.sort_values('prediction', ascending=False).head(top_n)
                results = []
                for _, row in pred_df.iterrows():
                    item_idx = int(row['item_idx'])
                    waste_id, application = self.idx_to_item[item_idx]
                    waste_info = self.waste_df[self.waste_df['waste_id'] == waste_id].iloc[0]
                    results.append({
                        'waste_id': waste_id,
                        'waste_name': waste_info['waste_name'],
                        'application': application,
                        'confidence': min(row['prediction'] / 5.0, 1.0),
                        'material_category': waste_info.get('material_category', 'Organic'),
                        'processing_difficulty': waste_info.get('processing_difficulty', 'Medium')
                    })
                return results
            else:
                return self.get_popular_recommendations(top_n)
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return self.get_popular_recommendations(top_n)

    def get_popular_recommendations(self, top_n=5):
        """Get most popular items based on average ratings."""
        try:
            item_means = self.user_item_matrix.mean().sort_values(ascending=False)
            top_items = item_means.head(top_n)
            results = []
            for item_id, mean_rating in top_items.items():
                item_id = int(item_id)
                waste_id, application = self.idx_to_item[item_id]
                waste_info = self.waste_df[self.waste_df['waste_id'] == waste_id].iloc[0]
                results.append({
                    'waste_id': waste_id,
                    'waste_name': waste_info['waste_name'],
                    'application': application,
                    'confidence': mean_rating / 5.0,
                    'material_category': waste_info.get('material_category', 'Organic'),
                    'processing_difficulty': waste_info.get('processing_difficulty', 'Medium')
                })
            return results
        except Exception as e:
            logger.error(f"Error getting popular recommendations: {str(e)}")
            return []

    def add_rating(self, user_id, waste_id, application, rating):
        """Add a new rating to the system."""
        try:
            item_key = (waste_id, application)
            if item_key not in self.item_to_idx:
                new_item_id = len(self.items_df)
                self.items_df.loc[new_item_id] = [waste_id, application]
                self.item_to_idx[item_key] = new_item_id
                self.idx_to_item[new_item_id] = item_key
                self.user_item_matrix[new_item_id] = np.nan
            if user_id not in self.user_to_idx:
                new_user_idx = len(self.user_to_idx)
                self.user_to_idx[user_id] = new_user_idx
                self.idx_to_user[new_user_idx] = user_id
                new_row = pd.Series(np.nan, index=self.user_item_matrix.columns)
                self.user_item_matrix.loc[user_id] = new_row
            item_idx = self.item_to_idx[item_key]
            self.user_item_matrix.loc[user_id, item_idx] = rating
            new_rating = pd.DataFrame({
                'user_id': [user_id],
                'waste_id': [waste_id],
                'application': [application],
                'rating': [rating],
                'timestamp': [pd.Timestamp.now()]
            })
            self.ratings_df = pd.concat([self.ratings_df, new_rating], ignore_index=True)
            if (self.user_item_matrix.shape[0] >= self.min_ratings and
                    self.user_item_matrix.shape[1] >= self.min_ratings):
                self.initialize_model()
            return True
        except Exception as e:
            logger.error(f"Error adding rating: {str(e)}")
            return False