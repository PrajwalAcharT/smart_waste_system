from collections import defaultdict
import logging
import json
import os
import pandas as pd

logger = logging.getLogger("HybridRecommender")


class HybridRecommender:
    def __init__(self, content_rec, collab_rec, waste_df):
        self.content = content_rec
        self.collab = collab_rec
        self.waste_df = waste_df
        self.waste_cache = waste_df.set_index('waste_id').to_dict('index')
        self.application_details = self._load_application_details()

    def _load_application_details(self):
        """Load detailed information about applications from JSON file."""
        try:
            if os.path.exists("data/application_details.json"):
                with open("data/application_details.json", "r") as f:
                    return json.load(f)
            else:
                details = self._generate_default_details()
                with open("data/application_details.json", "w") as f:
                    json.dump(details, f, indent=2)
                return details
        except Exception as e:
            logger.error(f"Failed to load application details: {str(e)}")
            return self._generate_default_details()

    def _generate_default_details(self):
        """Generate default application details when file is missing."""
        return {
            "Compost": {
                "steps": ["Collect crop waste and shred into smaller pieces"],
                "safety": ["Wear gloves when handling decomposing materials"],
                "equipment": ["Compost bin or dedicated area"],
                "roi_timeline": "Medium",
                "investment_level": "Low"
            },
            "Biogas": {
                "steps": ["Construct or purchase biogas digester system"],
                "safety": ["Ensure proper gas sealing to prevent leaks"],
                "equipment": ["Biogas digester"],
                "roi_timeline": "Long",
                "investment_level": "High"
            }
        }

    def recommend(self, user_id=None, waste_id=None, top_n=5, content_weight=0.6, collab_weight=0.4):
        """Generate hybrid recommendations combining content-based and collaborative filtering."""
        content_recs = self.content.get_recommendations(waste_id, top_n=top_n * 2) if waste_id else []
        collab_recs = self.collab.recommend_for_user(user_id, top_n=top_n * 2) if user_id else []
        if not collab_recs and content_recs:
            content_weight = 1.0
            collab_weight = 0.0
        merged_scores = defaultdict(float)
        for rec in content_recs:
            key = (rec['waste_id'], rec['application'])
            merged_scores[key] += rec['similarity_score'] * content_weight * 100
        for rec in collab_recs:
            key = (rec['waste_id'], rec['application'])
            merged_scores[key] += rec['confidence'] * collab_weight * 20
        results = []
        for (w_id, app), score in sorted(merged_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]:
            waste_data = self.waste_cache.get(w_id, {})
            if not waste_data:
                continue
            results.append({
                'waste_id': w_id,
                'waste_name': waste_data.get('waste_name', 'Unknown'),
                'application': app,
                'score': score,
                'material_category': waste_data.get('material_category', 'Organic'),
                'processing_difficulty': waste_data.get('processing_difficulty', 'Medium')
            })
        return results

    def get_application_details(self, waste_id, application):
        """Get detailed implementation steps for a specific application."""
        app_details = self.application_details.get(application, {})
        if not app_details:
            return None
        waste_data = self.waste_cache.get(waste_id, {})
        if not waste_data:
            return None
        return {
            'waste_name': waste_data.get('waste_name', 'Unknown'),
            'waste_id': waste_id,
            'application': application,
            'steps': app_details.get('steps', []),
            'safety': app_details.get('safety', []),
            'equipment': app_details.get('equipment', [])
        }