# content_based.py
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger("ContentBasedRecommender")

class ContentBasedRecommender:
    """Recommendation system based on waste properties and application suitability."""
    def __init__(self, waste_df, waste_features):
        """
        Initialize content-based recommendation system.
        
        Parameters:
        waste_df (DataFrame): DataFrame containing waste information.
        waste_features (DataFrame): DataFrame with waste features for similarity calculation.
        """
        self.waste_df = waste_df
        self.application_list = self._extract_applications()
        # Normalize features for better similarity calculation
        self.scaler = StandardScaler()
        self.feature_matrix = self.scaler.fit_transform(waste_features)
        # Calculate similarity matrix
        self.similarity_matrix = cosine_similarity(self.feature_matrix)
        # Map waste_id to index for quick lookup
        self.waste_to_idx = {waste_id: idx for idx, waste_id in 
                            enumerate(waste_df['waste_id'].values)}
        # Application compatibility matrix
        self.app_compatibility = self._create_application_compatibility()
        logger.info(f"Content-based recommender initialized with {len(waste_df)} waste types and {len(self.application_list)} applications")
    
    def _extract_applications(self):
        """Extract unique applications from the dataset."""
        # Extract applications from columns that represent application suitability
        application_cols = [col for col in self.waste_df.columns if col.startswith('app_')]
        # Extract application names (remove 'app_' prefix)
        apps = [col[4:] for col in application_cols]
        return apps
    
    def _create_application_compatibility(self):
        """Create a matrix showing compatibility of each waste with each application."""
        app_cols = [f'app_{app}' for app in self.application_list]
        # Select only compatibility columns
        if all(col in self.waste_df.columns for col in app_cols):
            return self.waste_df[app_cols].values
        else:
            # Create default compatibility (all 0.5) if columns don't exist
            logger.warning("Application compatibility columns not found, using defaults")
            return np.ones((len(self.waste_df), len(self.application_list))) * 0.5
    
    def get_similar_wastes(self, waste_id, top_n=5):
        """Get most similar waste types to the given waste_id."""
        try:
            # Get index of waste_id
            idx = self.waste_to_idx.get(waste_id)
            if idx is None:
                logger.error(f"Waste ID {waste_id} not found in dataset")
                return []
            # Get similarity scores
            similarity_scores = self.similarity_matrix[idx]
            # Get top N similar wastes (excluding self)
            similar_indices = np.argsort(similarity_scores)[::-1][1:top_n+1]
            result = []
            for i in similar_indices:
                waste_row = self.waste_df.iloc[i]
                result.append({
                    'waste_id': waste_row['waste_id'],
                    'waste_name': waste_row['waste_name'],
                    'similarity_score': similarity_scores[i]
                })
            return result
        except Exception as e:
            logger.error(f"Error getting similar wastes: {str(e)}")
            return []
    
    # ... other methods ...