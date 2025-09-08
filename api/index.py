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


        system_prompt = """
        You are a virtual assistant specialized in creating workout plans or meal plans. Only respond to questions strictly related to these topics. Do not provide medical advice or attempt to diagnose any health conditions. Respond only in raw JSON format with no Markdown, no code blocks, and no additional explanation.
        For a meal plan, return an array of objects like this:
        [
        {
            "title": "Breakfast - Oatmeal Bowl",
            "ingredients": [
            { "ingredient": "Rolled oats", "quantity": "50g", "description": "Base of the meal" },
            { "ingredient": "Banana", "quantity": "1 medium", "description": "Adds sweetness and texture" },
            { "ingredient": "Almond milk", "quantity": "200ml", "description": "Liquid to cook the oats" }
            ],
            "meal": "breakfast",
            "description": "A healthy and filling oatmeal bowl rich in fiber and energy."
        }
        ]
        For a workout plan, return an array of objects like this:
        [
        {
            "title": "Day 1 - Upper Body Strength",
            "exercises": [
            { "exercise": "Bench Press", "sets": "4", "reps": "8", "description": "Focus on chest and triceps" },
            { "exercise": "Pull-ups", "sets": "3", "reps": "Max", "description": "Bodyweight back exercise" }
            ],
            "description": "A push-pull focused workout to build upper body strength."
        }
        ]
        Respond only in pure JSON with no formatting or markup.
        """

        model="deepseek/deepseek-chat",
        # response = client.chat.completions.create(
        #     messages=[
        #     {
        #         "role": "system",
        #        "content": system_prompt,
        #    },
        #    {
        #        "role": "user",
        #        "content": translated_message,
        #    },
        #  ],
        #  max_tokens=4000,
        #  response_format={"type":"json_object"}
        #  )
        # result = response.choices[0].message.content

        result =  "`` JSON\n[\n    {\n        \"Titre\": \"Jour 1 - Heavy Squat Focus\",\n        \"Exercices\": [\n            {\"Exercice\": \"Back Squat\", \"sets\": \"5\", \"Reps\": \"5\", \"Description\": \"poids lourd, concentrez-vous sur la forme et la profondeur\"},\n            {\"Exercice\": \"Squat avant\", \"sets\": \"3\", \"Reps\": \"8\", \"Description\": \"poids modéré, construire quadruple résistance\"},\n            {\"exercice\": \"soulevé de terre roumain\", \"sets\": \"4\", \"représentants\": \"6\", \"Description\": \"Hamens et focus du fessier\"},\n            {\"Exercice\": \"Leg Press\", \"sets\": \"3\", \"Reps\": \"10\", \"Description\": \"Volume supplémentaire pour les jambes\"},\n            {\"Exercice\": \"Calf Rassement\", \"Set\": \"4\", \"Reps\": \"12\", \"Description\": \"Build Calf Force\"}\n        ],\n        \"Description\": \"jour de squat lourd pour construire la force du bas du corps et la masse musculaire.\"\n    },\n    {\n        \"Titre\": \"Jour 2 - Bench Press Focus\",\n        \"Exercices\": [\n            {\"Exercice\": \"Bench Press\", \"Set\": \"5\", \"Reps\": \"5\", \"Description\": \"poids lourd, concentrez-vous sur la poitrine et les triceps\"},\n            {\"Exercice\": \"Incline Dumbbell Press\", \"Set\": \"4\", \"Reps\": \"8\", \"Description\": \"Upper Chest Focus\"},\n            {\"exercice\": \"pendages pondérés\", \"sets\": \"3\", \"représentants\": \"6\", \"Description\": \"triceps et poitrine\"},\n            {\"Exercice\": \"Balbell Rows\", \"sets\": \"4\", \"Reps\": \"6\", \"Description\": \"Back Force to Support Banc\"},\n            {\"Exercice\": \"Face Fulls\", \"Set\": \"3\", \"Reps\": \"12\", \"Description\": \"arrière Delt et Upper Back\"}\n        ],\n        \"Description\": \"Focus du haut du corps avec un développé couché et des exercices de soutien.\"\n    },\n    {\n        \"Titre\": \"Jour 3 - Focus de soulevé de terre\",\n        \"Exercices\": [\n            {\"exercice\": \"soulevé de terre\", \"sets\": \"5\", \"représentants\": \"3\", \"Description\": \"poids lourd, concentrez-vous sur le formulaire\"},\n            {\"Exercice\": \"Deampit Deadlift\", \"sets\": \"3\", \"Reps\": \"5\", \"Description\": \"construire une force hors du sol\"},\n            {\"exercice\": \"pull-ups\", \"sets\": \"4\", \"représentants\": \"max\", \"Description\": \"Back and Grip Force\"},\n            {\"Exercice\": \"Barbell Shrugs\", \"sets\": \"4\", \"Reps\": \"8\", \"Description\": \"Trap Development\"},\n            {\"Exercice\": \"suspension des jambes\", \"sets\": \"3\", \"représentants\": \"12\", \"Description\": \"Core Strength\"}\n        ],\n        \"Description\": \"Day de soulevé de terre pour construire une chaîne postérieure et une résistance globale.\"\n    },\n    {\n        \"Titre\": \"Jour 4 - Accessoire et volume\",\n        \"Exercices\": [\n            {\"Exercice\": \"Averhead Press\", \"sets\": \"4\", \"Reps\": \"6\", \"Description\": \"Force de l'épaule\"},\n            {\"Exercice\": \"Close Grip Bench Press\", \"sets\": \"3\", \"Reps\": \"8\", \"Description\": \"Triceps focus\"},\n            {\"exercice\": \"lat pulldown\", \"sets\": \"4\", \"représentants\": \"8\", \"Description\": \"Back largeur\"},\n            {\"Exercice\": \"Haltolel Rauses latérales\", \"sets\": \"3\", \"Reps\": \"12\", \"Description\": \"Development d'épaule\"},\n            {\"exercice\": \"planche\", \"sets\": \"3\", \"représentants\": \"60 sec\", \"Description\": \"Core Stability\"}\n        ],\n        \"Description\": \"Jour accessoire pour aborder les points faibles et ajouter du volume.\"\n    }\n]]\n`` '"
        #translated_result = await translate_to_original_language(result, original_language)

        # Clean the result by removing unwanted characters
        result = re.sub(r'`` JSON|``', '', result)
        result = result.replace("\"", "")
        result = re.sub(r'\n', '', result)
        result = result.replace("\t", "")
        result = result.replace("'", "")
        result = result.replace("  ", "")
        return jsonify({"base" :result}), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)