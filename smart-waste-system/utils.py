import pandas as pd
import numpy as np
import logging
import shutil
import os
from datetime import datetime
import asyncio

logger = logging.getLogger("Utilities")


def validate_image(image):
    """
    Validate uploaded image file.
    
    Parameters:
        image: The uploaded image object (e.g., from PIL or similar library).
    
    Raises:
        ValueError: If the image fails validation checks.
    
    Returns:
        bool: True if validation passes.
    """
    try:
        # Check file size (10MB limit)
        if hasattr(image, "size") and image.size > 10 * 1024 * 1024:  # 10MB limit
            raise ValueError("Image size exceeds 10MB limit.")
        
        # Check file format
        if hasattr(image, "format") and image.format not in ['JPEG', 'PNG']:
            raise ValueError("Only JPEG/PNG formats are supported.")
        
        logger.info("Image validation passed.")
        return True
    except Exception as e:
        logger.error(f"Image validation failed: {str(e)}")
        raise


async def async_predict(model, image):
    """
    Perform asynchronous model prediction.
    
    Parameters:
        model: The trained machine learning model.
        image: The input image for prediction.
    
    Returns:
        Prediction result from the model.
    """
    try:
        loop = asyncio.get_event_loop()
        logger.info("Starting asynchronous model prediction...")
        result = await loop.run_in_executor(None, model.predict, image)
        logger.info("Asynchronous prediction completed.")
        return result
    except Exception as e:
        logger.error(f"Asynchronous prediction failed: {str(e)}")
        raise


def backup_data():
    """
    Create a timestamped data backup of critical files.
    
    Creates a backup directory with the current timestamp and copies essential files.
    """
    try:
        backup_dir = os.path.join("backups", datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(backup_dir, exist_ok=True)
        
        # List of files to back up
        files_to_backup = [
            "data/waste_types.csv",
            "data/user_ratings.csv"
        ]
        
        for file_path in files_to_backup:
            if os.path.exists(file_path):
                shutil.copy(file_path, backup_dir)
                logger.info(f"Copied {file_path} to {backup_dir}")
            else:
                logger.warning(f"File {file_path} not found, skipping backup.")
        
        logger.info(f"Backup created successfully in {backup_dir}")
    except Exception as e:
        logger.error(f"Backup process failed: {str(e)}")


def log_feedback(user_id, rating, comments=""):
    """
    Log user feedback to a log file.
    
    Parameters:
        user_id (str): Unique identifier for the user.
        rating (int): User-provided rating (e.g., 1-5).
        comments (str): Optional comments from the user.
    """
    try:
        log_file = "logs/feedback.log"
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        with open(log_file, "a") as f:
            timestamp = datetime.now().isoformat()
            f.write(f"{timestamp},{user_id},{rating},{comments}\n")
        
        logger.info(f"Feedback logged successfully for user {user_id}.")
    except Exception as e:
        logger.error(f"Failed to log feedback: {str(e)}")


def load_feedback() -> pd.DataFrame:
    """
    Load feedback data from the log file for analysis.
    
    Returns:
        pd.DataFrame: DataFrame containing feedback data, or an empty DataFrame if no data exists.
    """
    try:
        log_file = "logs/feedback.log"
        if os.path.exists(log_file):
            feedback_df = pd.read_csv(
                log_file,
                names=['timestamp', 'user_id', 'rating', 'comments']
            )
            logger.info(f"Loaded {len(feedback_df)} feedback entries.")
            return feedback_df
        else:
            logger.warning("Feedback log file not found, returning empty DataFrame.")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Failed to load feedback data: {str(e)}")
        return pd.DataFrame()


def clear_old_backups(days=30):
    """
    Delete old backup directories older than a specified number of days.
    
    Parameters:
        days (int): Number of days after which backups should be deleted (default: 30).
    """
    try:
        backup_dir = "backups"
        if not os.path.exists(backup_dir):
            logger.info("No backup directory found, skipping cleanup.")
            return
        
        cutoff_date = datetime.now() - pd.Timedelta(days=days)
        deleted_count = 0
        
        for dir_name in os.listdir(backup_dir):
            dir_path = os.path.join(backup_dir, dir_name)
            try:
                # Parse directory name as a timestamp
                dir_timestamp = datetime.strptime(dir_name, "%Y%m%d_%H%M%S")
                if dir_timestamp < cutoff_date:
                    shutil.rmtree(dir_path)
                    deleted_count += 1
                    logger.info(f"Deleted old backup: {dir_path}")
            except ValueError:
                logger.warning(f"Skipping invalid backup folder: {dir_path}")
            except Exception as e:
                logger.error(f"Failed to delete backup {dir_path}: {str(e)}")
        
        logger.info(f"Deleted {deleted_count} old backups older than {days} days.")
    except Exception as e:
        logger.error(f"Backup cleanup failed: {str(e)}")