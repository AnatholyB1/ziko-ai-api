from flask import Flask, request, jsonify, send_from_directory
import tensorflow as tf
from flask_cors import CORS
from pytubefix import YouTube
import cv2
from collections import deque
import numpy as np
import os
from openai import OpenAI
from googletrans import Translator
import asyncio
import re
import json


translator = Translator()

app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/')
def home():
    return 'Hello, World!'

@app.route('/about')
def about():
    return 'About'


@app.route('/videos/<path:filename>')
def serve_video(filename):
    return send_from_directory('../videos', filename)

model_path = 'models/LRCN_model__Date_Time_2024_11_08__22_46_09__Loss_0.41232773661613464__Accuracy_0.868852436542511.h5'

if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
else:
    print(f"Model file not found at {model_path}")
    model = None  # ou gérer l\'absence du modèle de manière appropriée

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
           
        video_url = f'http://{request.host}/videos/{video_title}.mp4'   
        video_reader.release()
        return jsonify({'prediction': predicted_class_name, "confidence": confidence,  "video_url": video_url}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
   
client = OpenAI(
    base_url="https://api.aimlapi.com/v1",
    # Insert your AIML API Key in the quotation marks instead of <YOUR_API_KEY>.
    api_key=os.getenv("AIML_API_KEY") ,  
)


 
async def translate_to_english(text):
    # Détecter la langue du texte
    detected =  await translator.detect(text)
    original_language = detected.lang
    # Traduire le texte en anglais
    translated =  await translator.translate(text, dest='en')
    return translated.text, original_language

async def translate_to_original_language(text, original_language):
    # Traduire le texte dans la langue d\'origine
    translated =  await translator.translate(text, dest=original_language)
    return translated.text


  
  
@app.route('/chat', methods=['POST'])
async def chat():
    try:
        data = request.get_json()
        message = data['message']
        if not message:
            return jsonify({'error': "message is required"}), 400
        
        translated_message, original_language  =  await translate_to_english(message)

        response = client.chat.completions.create(
        model="deepseek/deepseek-chat",
        messages=[
            {
                 "role": "system",
               "content": "You are a virtual assistant specialized in creating workout plans or meal plans. Only respond to questions related to these topics. Do not provide medical advice or diagnose health conditions.  If it's a meal plan i want an array of object [{ title, {ingredient, quantity, description}, meal, description}] if it's a workout plan i want an array of object [{title, {exercise, sets, reps, description}, description}]",
           },
           {
               "role": "user",
               "content": translated_message,
           },
         ],
         max_tokens=1000,
         response_format={"type":"text"}
         )

        result = response.choices[0].message.content
        return jsonify({ "base" :result }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)