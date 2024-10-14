import moviepy.editor as mp
import random
import os
import pandas as pd

#%%

def resize_clip(clip, target_resolution):
    return clip.resize(target_resolution)

def process_videos_and_image(video_paths, image_path, output_resolution,order_matrix):
    # Load and resize the image
    image_clip = mp.ImageClip(image_path).set_duration(5)
    image_clip = resize_clip(image_clip, output_resolution)

    # Shuffle the videos in a random order
    random.shuffle(video_paths)
    print(f'Orden: {video_paths}')
    order_matrix = pd.concat([order_matrix,pd.Series(video_paths)],axis=1)
    
    # Process and resize all the videos
    video_clips = []
    for video in video_paths:
        try:
            clip = mp.VideoFileClip(video)
            clip = resize_clip(clip, output_resolution)
            video_clips.append(clip)
            print(f"Video processed successfully: {video}")
        except Exception as e:
            print(f"Failed to process video: {video}, Error: {e}")

    if not video_clips:
        print("No videos processed successfully.")
        return None, None

    return video_clips, image_clip, order_matrix

def concatenate_videos_with_image(video_paths, image_path, output_resolution, output_path,order_matrix):
    video_clips, image_clip, order_matrix = process_videos_and_image(video_paths, image_path, output_resolution,order_matrix)
    
    if video_clips is None:
        return


    # Concatenate the videos with the image in between
    final_clips = []
    for i, video_clip in enumerate(video_clips):
        final_clips.append(video_clip)
        if i < len(video_clips) - 1:  # Add the image between videos
            final_clips.append(image_clip)

    # Combine all clips into a final video
    final_video = mp.concatenate_videoclips(final_clips)

    # Write the final video to a file
    final_video.write_videofile(output_path, codec="libx264")
    
    return order_matrix

def get_video_files(path):
    dir_files = os.listdir(base_dir)
    video_files = [f'{base_dir}/{file}' for file in dir_files if file[-3:]=="mp4"]
    
    return video_files

#%% 
base_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(f'{base_dir}/output_videos', exist_ok=True)

video_files = get_video_files(base_dir)
image_file = f'{base_dir}/image.png'
output_resolution = (1280, 720)  # Target resolution (width, height)
order_matrix = pd.DataFrame()

# Change range to get n output_videos
for n in range(2):
    output_file = f'{base_dir}/output_videos/output_video-{n}.mp4'
    order_matrix = concatenate_videos_with_image(video_files, image_file, output_resolution, output_file, order_matrix)

order_matrix.to_excel(f'{base_dir}/order_matrix.xlsx')