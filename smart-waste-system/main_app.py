import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import pandas as pd
import time
import logging
import matplotlib.pyplot as plt
from hybrid_classifier import HybridCropClassifier
from data_preprocessing import load_and_process
from hybrid_recommender import HybridRecommender
from collaborative_filtering import CollaborativeRecommender
from content_based import ContentBasedRecommender
import pathlib
import os

# Configure logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(os.path.join(log_dir, 'app.log')), logging.StreamHandler()]
)
logger = logging.getLogger("AgriWasteAdvisor")

# Configuration
CLASS_NAMES = ['bajra', 'castor', 'cotton', 'paddy', 'sugarcane', 'wheat']
MODEL_DIR = "models"
DATA_PATH = "data"

@st.cache_resource
def initialize_systems():
    """Initialize all components with caching and error handling"""
    try:
        logger.info("Initializing systems...")
        # Initialize classifier
        classifier = HybridCropClassifier.load_latest_model()
        # Initialize recommender system
        waste_df, ratings_df, waste_features, user_item_matrix = load_and_process()
        content_rec = ContentBasedRecommender(waste_df, waste_features)
        collab_rec = CollaborativeRecommender(user_item_matrix, ratings_df, waste_df)
        hybrid_rec = HybridRecommender(content_rec, collab_rec, waste_df)
        logger.info("System initialization completed")
        return classifier, hybrid_rec, waste_df
    except Exception as e:
        logger.error(f"Initialization failed: {str(e)}")
        st.error("System initialization failed. Please check logs.")
        st.stop()

def main():
    st.set_page_config(
        page_title="AI-Powered Crop Waste Reuse Advisor", 
        page_icon="🌾", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    # Initialize systems
    classifier, recommender, waste_df = initialize_systems()
    # Session state management
    if 'current_crop' not in st.session_state:
        st.session_state.current_crop = None
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = []
    if 'user_id' not in st.session_state:
        st.session_state.user_id = "user_" + str(int(time.time()))
    if 'previous_ratings' not in st.session_state:
        st.session_state.previous_ratings = {}

    # Main UI
    st.title("🌾 AI-Powered Crop Waste Reuse Advisor")
    st.markdown("Transform agricultural waste into valuable resources with AI assistance")
    st.markdown("---")

    # Sidebar User Management
    with st.sidebar:
        st.title("User Profile")
        farm_size = st.selectbox("Farm Size", ["Small (< 5 acres)", "Medium (5-20 acres)", "Large (> 20 acres)"])
        farm_type = st.multiselect("Farming Type", ["Organic", "Traditional", "Mixed", "Sustainable"], default=["Traditional"])
        region = st.selectbox("Agricultural Region", ["North", "South", "East", "West", "Central"])
        st.markdown("---")
        st.subheader("System Status")
        st.markdown(f"**Last Classification:** {st.session_state.current_crop.title() if st.session_state.current_crop else 'None'}")
        st.markdown(f"**User ID:** {st.session_state.user_id}")
        system_health = classifier.model_confidence() if hasattr(classifier, 'model_confidence') else 0.85
        st.progress(system_health, text=f"System Health: {system_health*100:.0f}%")

    # Main Workflow Tabs
    tab1, tab2, tab3 = st.tabs(["📷 Crop Identification", "♻️ Waste Reuse Recommendations", "📊 Sustainability Impact"])

    # Tab 1: Crop Identification
    with tab1:
        st.header("Crop Identification")
        st.write("Upload or capture an image of your crop for automated identification")
        col1, col2 = st.columns([1, 1.5])
        with col1:
            upload_option = st.radio("Input Method", ["Upload Image", "Use Camera"], horizontal=True)
            if upload_option == "Upload Image":
                img_file = st.file_uploader(
                    "Upload crop image", 
                    type=["jpg", "png", "jpeg"],
                    help="Supported formats: JPEG, PNG"
                )
            else:
                img_file = st.camera_input("Capture live crop image")
        if img_file:
            try:
                image = Image.open(img_file)
                with col1:
                    st.image(image, caption="Uploaded Image", width=300)
                with col2:
                    with st.spinner("Analyzing crop..."):
                        start_time = time.time()
                        pred = classifier.predict(image)
                        class_idx = np.argmax(pred)
                        crop_type = CLASS_NAMES[class_idx]
                        confidence = pred[class_idx]
                        duration = time.time() - start_time
                        logger.info(f"Classification: {crop_type} ({confidence:.2f}) in {duration:.2f}s")
                    st.success(f"""
                    **Identification Result**  
                    Crop Type: **{crop_type.title()}**  
                    Confidence: **{confidence*100:.1f}%**  
                    Processing Time: {duration:.2f}s
                    """)
                    # Visualization
                    fig, ax = plt.subplots()
                    sorted_indices = np.argsort(pred)[::-1]
                    sorted_preds = [pred[i] for i in sorted_indices]
                    sorted_names = [CLASS_NAMES[i].title() for i in sorted_indices]
                    bars = ax.barh(sorted_names, sorted_preds, color='skyblue')
                    ax.set_xlabel('Confidence')
                    ax.set_title('Classification Results')
                    # Add percentage annotations
                    for i, bar in enumerate(bars):
                        width = bar.get_width()
                        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                                f'{width*100:.1f}%', va='center')
                    st.pyplot(fig)
                    st.session_state.current_crop = crop_type
                    # Auto-navigate to recommendations
                    if confidence > 0.7:
                        st.info("High confidence detection! See the 'Waste Reuse Recommendations' tab for reuse options.")
            except Exception as e:
                logger.error(f"Classification error: {str(e)}")
                st.error("Error processing image. Please try again.")

    # Tab 2: Recommendations
    with tab2:
        st.header("Waste Reuse Recommendations")
        if not st.session_state.current_crop:
            st.info("Please identify a crop first in the 'Crop Identification' tab")
        else:
            st.write(f"Showing reuse options for **{st.session_state.current_crop.title()}** agricultural waste")
            # Filter options
            col1, col2, col3 = st.columns(3)
            with col1:
                difficulty = st.select_slider(
                    "Processing Difficulty",
                    options=["Easy", "Moderate", "Advanced"],
                    value="Moderate"
                )
            with col2:
                investment = st.select_slider(
                    "Investment Level",
                    options=["Low", "Medium", "High"],
                    value="Medium"
                )
            with col3:
                timeline = st.select_slider(
                    "Implementation Timeline",
                    options=["Quick (Days)", "Medium (Weeks)", "Long (Months)"],
                    value="Medium (Weeks)"
                )
            try:
                waste_id = waste_df[waste_df['waste_name'] == st.session_state.current_crop]['waste_id'].values[0]
                with st.spinner("Generating personalized recommendations..."):
                    start_time = time.time()
                    # Set weights based on user preferences
                    content_weight = 0.7 if farm_type and "Sustainable" in farm_type else 0.6
                    collab_weight = 1 - content_weight
                    recs = recommender.recommend(
                        user_id=st.session_state.user_id,
                        waste_id=waste_id,
                        content_weight=content_weight,
                        collab_weight=collab_weight,
                        top_n=6
                    )
                    duration = time.time() - start_time
                    # Filter based on user selections
                    difficulty_map = {"Easy": "Low", "Moderate": "Medium", "Advanced": "High"}
                    recs = [r for r in recs if r.get('processing_difficulty', 'Medium') == difficulty_map.get(difficulty, 'Medium')]
                    logger.info(f"Generated {len(recs)} recommendations in {duration:.2f}s")
                    st.session_state.recommendations = recs
                # Display recommendations
                if not recs:
                    st.warning("No recommendations match your current filters. Try adjusting your preferences.")
                else:
                    st.subheader("Top Reuse Options")
                    # Display cards in grid
                    cols = st.columns(3)
                    for idx, rec in enumerate(recs[:6]):
                        col_idx = idx % 3
                        with cols[col_idx]:
                            with st.container(border=True):
                                st.markdown(f"#### {rec['application']}")
                                st.write(f"**Material:** {rec['material_category']}")
                                st.write(f"**Difficulty:** {rec['processing_difficulty']}")
                                st.write(f"**ROI Timeline:** {'Short' if rec['score'] > 80 else 'Medium'}")
                                st.progress(min(rec['score']/100, 1.0), text=f"Match Score: {rec['score']:.1f}%")
                                # User rating
                                rating_key = f"{rec['waste_id']}_{rec['application']}"
                                if rating_key in st.session_state.previous_ratings:
                                    current_rating = st.session_state.previous_ratings[rating_key]
                                else:
                                    current_rating = 3
                                new_rating = st.slider(
                                    "Rate this recommendation", 
                                    1, 5, 
                                    current_rating,
                                    key=f"rating_{rating_key}"
                                )
                                if new_rating != current_rating:
                                    st.session_state.previous_ratings[rating_key] = new_rating
                                    # Log rating for collaborative filtering
                                    with open("logs/feedback.log", "a") as f:
                                        f.write(f"{time.time()},{st.session_state.user_id},{rec['waste_id']},{rec['application']},{new_rating}\n")
                                    st.success("Feedback saved!")
                    # Detailed view
                    st.markdown("### Detailed Implementation Plan")
                    if recs:
                        selected_app = st.selectbox(
                            "Select application for detailed instructions",
                            [r['application'] for r in recs]
                        )
                        selected_rec = next(r for r in recs if r['application'] == selected_app)
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            st.subheader(f"{selected_rec['application']} Implementation")
                            steps = recommender.get_application_details(
                                selected_rec['waste_id'], 
                                selected_app
                            )
                            if not steps:
                                st.warning("Detailed implementation plan not available.")
                            else:
                                st.markdown("#### Process Steps")
                                for i, step in enumerate(steps.get('steps', [])):
                                    st.markdown(f"{i+1}. {step}")
                                st.markdown("#### Required Equipment")
                                equipment = [
                                    "Collection containers", 
                                    "Processing tools",
                                    "Safety equipment",
                                    "Storage containers"
                                ]
                                for item in equipment:
                                    st.markdown(f"- {item}")
                                st.markdown("#### Cost Estimation")
                                st.info(f"Estimated setup cost: $500-2000 depending on scale")
                                st.info(f"Potential ROI: 15-30% within first year")
                        with col2:
                            st.subheader("Safety Guidelines")
                            for guideline in steps.get('safety', ["Follow general safety protocols"]):
                                st.markdown(f"⚠️ {guideline}")
                            st.subheader("Sustainability Impact")
                            impact_score = min(selected_rec['score'] / 20, 5)
                            st.markdown(f"**Environmental Impact Score:** {impact_score:.1f}/5.0")
                            benefits = [
                                "Reduces waste sent to landfill",
                                "Lowers carbon footprint",
                                "Conserves natural resources",
                                "Potential additional income stream"
                            ]
                            st.markdown("**Benefits:**")
                            for benefit in benefits:
                                st.markdown(f"✅ {benefit}")
            except Exception as e:
                logger.error(f"Recommendation error: {str(e)}")
                st.error("Error generating recommendations. Please try again.")

    # Tab 3: Sustainability Impact
    with tab3:
        st.header("Sustainability Impact Analysis")
        if not st.session_state.current_crop:
            st.info("Please identify a crop first in the 'Crop Identification' tab")
        else:
            st.write(f"Analyzing sustainability impact for **{st.session_state.current_crop.title()}** waste management")
            # Set up placeholder metrics
            waste_volume = 5.0  # tons per acre
            total_acres = 10  # default acres
            # Allow user to input their specific details
            col1, col2 = st.columns(2)
            with col1:
                total_acres = st.number_input("Total Cultivated Area (acres)", min_value=1, value=10)
                waste_per_acre = st.slider("Waste Generated per Acre (tons)", 0.5, 10.0, 5.0, 0.5)
            with col2:
                current_disposal = st.radio(
                    "Current Disposal Method",
                    ["Open Burning", "Field Disposal", "Partial Reuse", "Landfill"]
                )
                reuse_target = st.slider("Target Waste Reuse (%)", 0, 100, 70, 5)
            # Calculate impact metrics
            total_waste = total_acres * waste_per_acre
            # Calculate carbon impact (simplified calculation)
            emission_factors = {
                "Open Burning": 1.5,  # tons CO2e per ton waste
                "Field Disposal": 0.8,
                "Partial Reuse": 0.5,
                "Landfill": 1.2
            }
            current_emissions = total_waste * emission_factors[current_disposal]
            potential_savings = current_emissions * (reuse_target / 100) * 0.8  # Assuming 80% efficiency
            # Financial impact (simplified)
            disposal_costs = {
                "Open Burning": 5,  # $ per ton
                "Field Disposal": 10,
                "Partial Reuse": 15,
                "Landfill": 25
            }
            current_cost = total_waste * disposal_costs[current_disposal]
            reuse_income = total_waste * (reuse_target / 100) * 20  # $20 per ton benefit
            financial_benefit = reuse_income + (current_cost * reuse_target / 100)  # Cost savings + income
            # Display impact metrics
            st.subheader("Impact Overview")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Total Waste Generated",
                    f"{total_waste:.1f} tons",
                    delta=None
                )
            with col2:
                st.metric(
                    "Carbon Emission Reduction",
                    f"{potential_savings:.1f} tons CO2e",
                    delta=f"{-potential_savings/current_emissions*100:.0f}%"
                )
            with col3:
                st.metric(
                    "Financial Benefit",
                    f"${financial_benefit:.0f}",
                    delta=f"${reuse_income:.0f} income"
                )
            # Visualization
            st.subheader("Environmental Impact Comparison")
            # Create comparison data
            methods = ["Current Practice", "With AI Advisor"]
            emissions = [current_emissions, current_emissions - potential_savings]
            costs = [current_cost, current_cost - (current_cost * reuse_target / 100)]
            # Plot comparisons
            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots()
                ax.bar(methods, emissions, color=['#ff9999', '#66b3ff'])
                ax.set_ylabel('CO2e Emissions (tons)')
                ax.set_title('Emissions Comparison')
                for i, v in enumerate(emissions):
                    ax.text(i, v + 0.1, f"{v:.1f}", ha='center')
                st.pyplot(fig)
            with col2:
                fig, ax = plt.subplots()
                ax.bar(methods, costs, color=['#ff9999', '#66b3ff'])
                ax.set_ylabel('Cost ($)')
                ax.set_title('Disposal Cost Comparison')
                for i, v in enumerate(costs):
                    ax.text(i, v + 0.1, f"${v:.0f}", ha='center')
                st.pyplot(fig)
            # Sustainability certification progress
            st.subheader("Sustainability Certification Progress")
            certification_score = min((reuse_target / 100) * 100, 100)
            st.progress(certification_score/100, text=f"Circular Economy Certification: {certification_score:.0f}%")
            threshold_descriptions = {
                30: "Basic Recycling Practices",
                60: "Advanced Waste Management",
                90: "Sustainable Agriculture Leader"
            }
            for threshold, description in threshold_descriptions.items():
                if certification_score >= threshold:
                    st.success(f"✅ {description} ({threshold}% threshold reached)")
                else:
                    st.info(f"⏳ {description} ({threshold-certification_score:.0f}% more needed)")

    # Feedback collection
    st.markdown("---")
    st.subheader("Advisor Feedback")
    with st.expander("Help us improve the AI Advisor"):
        feedback_text = st.text_area("Share your experience with the Crop Waste Reuse Advisor", 
                                    placeholder="What worked well? What could be improved?")
        overall_rating = st.slider("Rate your overall experience", 1, 5, 4)
        if st.button("Submit Feedback"):
            try:
                with open("logs/user_feedback.log", "a") as f:
                    f.write(f"{time.time()},{st.session_state.user_id},{overall_rating},{feedback_text.replace(',', ';')}\n")
                st.success("Thank you for your feedback! It helps us improve the advisor.")
            except Exception as e:
                logger.error(f"Feedback error: {str(e)}")
                st.error("Error saving feedback. Please try again.")

if __name__ == "__main__":
    main()