import os
videos = []
# Iterate over each file in the recorded_videos directory
base_dir = os.path.dirname(os.path.abspath(__file__))

# Define the directory where recorded videos will be saved
video_output_path = 'recorded_videos/'
os.makedirs(video_output_path, exist_ok=True)
for filename in os.listdir(video_output_path):
    if filename.endswith('.mp4'):
        # Construct the local path to each video file
        filepath = os.path.join(video_output_path, filename)
        # Append the filepath to the list of videos
        videos.append(filepath)
print(video_output_path)
print(videos)
