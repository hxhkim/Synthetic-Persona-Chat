import openai
import json
import os
import pprint
from dotenv import load_dotenv
from typing import Dict, List
import glob
import time

load_dotenv()

MODEL = "gpt-4o-mini"
openai.api_key = os.getenv("OPENAI_API_KEY")

client = openai

def label_emotion(data: Dict) -> Dict:
    for message in data['messages']:
        content = message.get('content', {})
        if not content.get('emotion_scores') and 'text' in content:
            text = content['text']
            prompt = f"""Analyze the following text and provide emotion_scores field for the following categories: Anger, Fear, Joy, Sadness, Surprise, Love, Boredom, Neutral. The output should be in JSON format with the emotion categories as keys and their respective scores as values, totaling exactly 100. 

            # Guidelines:
            1. Consider the Participant Information and Conversation Log when interpreting the emotional content of the text.
            2. Subtle emotional cues should be reflected in the corresponding emotion scores, but don't overinterpret.
            3. Assign a very high score to that emotion category when there is clear and strong evidence of a specific emotion in the text.

            # Participant Information: 
            {data["participant_persona"]},

            # Conversation Log:
            {data["messages"]}
        
            # Expected output example:
            "emotion_scores": {{
                "Anger": ,
                "Fear": ,
                "Joy": ,
                "Sadness": 
                "Surprise": ,
                "Love": ,
                "Boredom": ,
                "Neutral": 
            }},
            "text": "{text}"
            
            Ensure that your scoring reflects the intensity and clarity of the emotional expression in the text."""

            retries = 3
            for attempt in range(retries):
                try:
                    response = client.ChatCompletion.create(
                                model=MODEL,
                                messages=[
                                    {"role": "system", "content": "You are an AI assistant that helps to build conversation data set."},
                                    {"role": "user", "content": prompt}
                                ],
                                temperature=0.8,
                                response_format={"type": "json_object"}
                            )

                    response_content = response.choices[0].message.content

                    try:
                        filled_data = json.loads(response_content)  
                    except json.JSONDecodeError as e:
                        print(f"JSONDecodeError: {e}")
                        continue

                    if content.get('text') == text:
                        content['emotion_scores'] = filled_data.get('emotion_scores', {})

                    break  # Exit the retry loop if successful

                except openai.error.Timeout as e:
                    print(f"Attempt {attempt + 1} of {retries} failed with timeout. Retrying...")
                    time.sleep(3)  # Wait for 3 seconds before retrying
            else:
                print(f"Failed to process message: {text}")

    return data  # Return the data after processing all messages


# SPC-test

json_files = sorted(glob.glob('/home/user1/conversation-data/dataset-02-SPC/Synthetic-Persona-Chat/data/02_renamed_data/SPC-test/*.json'))

for i, json_file in enumerate(json_files[:5]):
    with open(json_file, 'r') as file:
        data = json.load(file)

    filled_data = label_emotion(data)
    print(f"filled_data_SPC-test_{i + 1}:", filled_data)
    print(f"Function end time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    print("=====================================")

    output_dir = '/home/user1/conversation-data/dataset-02-SPC/Synthetic-Persona-Chat/data/04_emotion-labeled_data/SPC-test'
    output_file = os.path.join(output_dir, f'emotion_labeled_data_SPC-test_{i + 1}.json')

    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(filled_data, file, ensure_ascii=False, indent=4)


# SPC-train

json_files = sorted(glob.glob('/home/user1/conversation-data/dataset-02-SPC/Synthetic-Persona-Chat/data/02_renamed_data/SPC-train/*.json'))

for i, json_file in enumerate(json_files[:5]):
    with open(json_file, 'r') as file:
        data = json.load(file)

    filled_data = label_emotion(data)
    print(f"filled_data_SPC-train_{i + 1}:", filled_data)
    print(f"Function end time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    print("=====================================")

    output_dir = '/home/user1/conversation-data/dataset-02-SPC/Synthetic-Persona-Chat/data/04_emotion-labeled_data/SPC-train'
    output_file = os.path.join(output_dir, f'emotion_labeled_data_SPC-train_{i + 1}.json')

    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(filled_data, file, ensure_ascii=False, indent=4)


# SPC-valid

json_files = sorted(glob.glob('/home/user1/conversation-data/dataset-02-SPC/Synthetic-Persona-Chat/data/02_renamed_data/SPC-valid/*.json'))

for i, json_file in enumerate(json_files[:5]):
    with open(json_file, 'r') as file:
        data = json.load(file)

    filled_data = label_emotion(data)
    print(f"filled_data_SPC-valid_{i + 1}:", filled_data)
    print(f"Function end time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    print("=====================================")

    output_dir = '/home/user1/conversation-data/dataset-02-SPC/Synthetic-Persona-Chat/data/04_emotion-labeled_data/SPC-valid'
    output_file = os.path.join(output_dir, f'emotion_labeled_data_SPC-valid_{i + 1}.json')

    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(filled_data, file, ensure_ascii=False, indent=4)


# New-SPC

json_files = sorted(glob.glob('/home/user1/conversation-data/dataset-02-SPC/Synthetic-Persona-Chat/data/02_renamed_data/New-SPC/*.json'))

for i, json_file in enumerate(json_files[:5]):
    with open(json_file, 'r') as file:
        data = json.load(file)

    filled_data = label_emotion(data)
    print(f"filled_data_New-SPC_{i + 1}:", filled_data)
    print(f"Function end time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    print("=====================================")

    output_dir = '/home/user1/conversation-data/dataset-02-SPC/Synthetic-Persona-Chat/data/04_emotion-labeled_data/New-SPC'
    output_file = os.path.join(output_dir, f'emotion_labeled_data_New-SPC_{i + 1}.json')

    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(filled_data, file, ensure_ascii=False, indent=4)