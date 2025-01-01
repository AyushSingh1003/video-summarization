import os
import torch
import torch.nn as nn
from models.feature_extractor import FeatureExtractor
from models.video_summarizer import VideoSummarizer
from models.text_summarizer import TextSummarizer
from utils.video_processing import extract_and_save_frames, preprocess_frames


def generate_video_summary(video_path, save_folder="Frames_data", interval=30):
    """
    Generates a video summary by extracting frames, processing them, and generating a summary.
    
    Args:
        video_path (str): Path to the input video.
        save_folder (str): Directory to save extracted frames.
        interval (int): Interval between frames to extract.
    
    Returns:
        str: Generated text summary of the video.
    """
    # Ensure save folder exists
    os.makedirs(save_folder, exist_ok=True)

    # Load models
    feature_extractor = FeatureExtractor().eval()
    summarizer = VideoSummarizer().eval()
    text_summarizer = TextSummarizer().eval()

    # Extract and preprocess frames
    frames = extract_and_save_frames(video_path, save_folder, interval)
    print(f"Extracted {len(frames)} frames.")
    
    frames_tensor = preprocess_frames(frames).float()  # Ensure float type
    print(f"Frames tensor shape: {frames_tensor.shape}, dtype: {frames_tensor.dtype}")

    # Extract frame features
    features = feature_extractor(frames_tensor.unsqueeze(0))  # Add batch dimension
    print(f"Features shape: {features.shape}")

    # Generate video summary embedding
    summary_embedding = summarizer(features)
    print(f"Summary embedding shape: {summary_embedding.shape}")

    # Ensure summary_embedding is of type LongTensor (integer type)
    summary_embedding = summary_embedding.long()  # Convert to LongTensor

    # Generate text summary
    text_summary = text_summarizer(summary_embedding)
    return text_summary


# Run summary generation on a test video
if __name__ == "__main__":
    video_path = "/Users/ayush/Downloads/video_summarization_Updated/data/test_video.mp4"
    save_folder = "Frames_data"  # Folder to save frames
    interval = 30  # Extract every 30th frame

    try:
        summary = generate_video_summary(video_path, save_folder, interval)
        print("Generated Summary:\n", summary)
    except Exception as e:
        print(f"Error: {e}")

