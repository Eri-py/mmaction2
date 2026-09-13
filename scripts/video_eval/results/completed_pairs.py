import csv
from pathlib import Path


def load_completed_pairs(output_path):
    """Return the set of (video_path, model_name) pairs already in the CSV."""
    output_path = Path(output_path)
    completed = set()
    if not output_path.exists():
        return completed
    with open(output_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            completed.add((row['video_path'], row['model_name']))
    return completed
