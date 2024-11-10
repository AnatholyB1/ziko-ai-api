from flask import Flask, request, jsonify
import tensorflow as tf
from pytubefix import YouTube
import cv2
from collections import deque
import numpy as np



app = Flask(__name__)



@app.route('/')
def home():
    return 'Hello, World!'

@app.route('/about')
def about():
    return 'About'

model_path = '../models/LRCN_model__Date_Time_2024_11_08__22_46_09__Loss_0.41232773661613464__Accuracy_0.868852436542511.h5'
model = tf.keras.models.load_model(model_path)

def download_youtube_videos(youtube_video_url, output_directory):
    try:

        # Extract video ID from YouTube Shorts URL
        if 'youtube.com/shorts/' in youtube_video_url:
            video_id = youtube_video_url.split('/')[-1]
            youtube_video_url = f'https://www.youtube.com/watch?v={video_id}'
            

        video = YouTube(youtube_video_url)
        title = "test"
        video_best = video.streams.get_highest_resolution()
        video_best.download(output_path=output_directory, filename=f'{title}.mp4')

        return title, None
    except Exception as e:
        print(f"Error downloading video: {e}")
        return None, str(e)






@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        youtube_video_url = data['youtube_video_url']

        directory = '../videos'



        video_title, error = download_youtube_videos(youtube_video_url, directory)
        if error:
            return jsonify({'error': error}), 500


        input_video_path = f'{directory}/{video_title}.mp4'
     

        Class_names = ["WalkinWithDog", "Taichi", "Swing", "HorseRAce"]

        image_height = 64
        image_width = 64

        sequence_length = 20

        video_reader = cv2.VideoCapture(input_video_path)

        frames_list = []

        video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

        skip_frames_window = max(int(video_frames_count / sequence_length), 1)

        predicted_class_name = ''

        for frame_counter in range(sequence_length):

            video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)

            success, frame = video_reader.read()
         

            if not success:
                return jsonify({'error': "could not read the video"}), 500

            resized_frame = cv2.resize(frame, (image_width, image_height))

            normalized_frame = resized_frame / 255

            frames_list.append(normalized_frame)

        

        predicted_labels_probabilities = model.predict(np.expand_dims(frames_list, axis=0))[0]
        
        predicted_label = np.argmax(predicted_labels_probabilities)

        predicted_class_name = Class_names[predicted_label]

                # Convert float32 to standard Python float
        confidence = float(predicted_labels_probabilities[predicted_label])
           
        video_reader.release()
        return jsonify({'prediction': predicted_class_name, "confidence": confidence}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
   
     


   


if __name__ == '__main__':
    app.run(debug=True)