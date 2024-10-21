from openai import OpenAI
from openai import RateLimitError
from openai import APITimeoutError
from openai import APIConnectionError
from openai import APIError
from openai import AuthenticationError

import json
import os
from typing import Dict, List
import re
import shutil
import time

MODEL = "gpt-4o-mini"
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

dataset = "SPC"
data_state = "emotion_labeled_data"
folder_name = "SPC-train"
input_directory_path = f"/home/ubuntu/conversation-data/dataset-02-SPC/Synthetic-Persona-Chat/data/03_filled_data/{folder_name}"
output_directory_path = f"/home/ubuntu/conversation-data/dataset-02-SPC/Synthetic-Persona-Chat/data/04_emotion_labeled_data/{folder_name}"
except_directory_path = f"/home/ubuntu/conversation-data/dataset-02-SPC/Synthetic-Persona-Chat/data/except"


def label_emotion(data: Dict, messages: List[Dict]) -> Dict:
    for message in data['messages']:
        content = message.get('content', {})
        if not content.get('emotion_scores') and 'text' in content:
            text = content['text']
            prompt = f"""Analyze the following text and provide emotion_scores field for the following categories: Anger, Fear, Joy, Sadness, Surprise, Love, Boredom, Neutral. The output should be in JSON format with the emotion categories as keys and their respective scores as values, totaling exactly 100. 

            # Guidelines:
            1. Consider the Participant Information and Conversation Log when interpreting the emotional content of the text.
            2. Subtle emotional cues should be reflected in the corresponding emotion scores, but don't overinterpret.
            3. Assign a very high score to that emotion category when there is clear and strong evidence of a specific emotion in the text.

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

            # Participant Information: 
            {data["participant_persona"]},

            # Conversation Log:
            {data["messages"]}
        
            Ensure that your scoring reflects the intensity and clarity of the emotional expression in the text."""


            # APITimeoutError
            retries = 3
            for attempt in range(retries):
                try:
                    start_time = time.time()
                    response = client.chat.completions.create(
                                model=MODEL,
                                messages=[
                                    {"role": "system", "content": "You are an AI assistant that helps to build conversation data set."},
                                    {"role": "user", "content": prompt}
                                ],
                                temperature=0.8,
                                response_format={"type": "json_object"}
                            )
                    end_time = time.time()
                    print(f"message: {text}")
                    print(f"API call took {end_time - start_time} seconds")
                    print("=====================================")

                    response_content = response.choices[0].message.content

                    try:
                        filled_data = json.loads(response_content)  
                    except json.JSONDecodeError as e:
                        print(f"JSONDecodeError: {e}")
                        continue

                    if content.get('text') == text:
                        content['emotion_scores'] = filled_data.get('emotion_scores', {})

                    break

                except (APITimeoutError, APIConnectionError, APIError, AuthenticationError) as e:
                    print(f"Attempt {attempt + 1} of {retries} failed with APITimeoutError. Retrying...")
                    time.sleep(3) 
            else:
                print(f"Failed to process message: {text}")

    return data  



def process_files(file_list: List[str], input_directory_path: str, output_directory_path: str):
    for filename in file_list:
        input_file_path = os.path.join(input_directory_path, filename)
        try:
            with open(input_file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            # Check if data is loaded correctly
            print(f"Processing file: {input_file_path}")
            print(f"Data: {data}")
            start_time = time.time()

            messages = data.get('messages', [])
            filled_data = label_emotion(data, messages)
            end_time = time.time()

            print(f"Processing {filename} took {end_time - start_time} seconds")
            print(f"emotion_labeled_data_{filename}:", filled_data)
            print("=====================================")

            # Extract the number from the original file name
            match = re.search(r'_(\d{5})\.json$', filename)
            if match:
                file_number = match.group(1)
            else:
                raise ValueError(f"Filename {filename} does not match the expected pattern")

            output_filename = f"{dataset}_{data_state}_{folder_name}_{file_number}.json"
            output_file_path = os.path.join(output_directory_path, output_filename)
            os.makedirs(output_directory_path, exist_ok=True)

            
            with open(output_file_path, 'w', encoding='utf-8') as file:
                json.dump(filled_data, file, ensure_ascii=False, indent=4)

        # RateLimitError
        except RateLimitError  as e:
            try:
                # retry logic
                retry_delay = 1
                max_retries = 5
                for retry in range(max_retries):
                    try:
                        with open(input_file_path, 'r', encoding='utf-8') as file:
                            data = json.load(file)
                        
                        # Check if data is loaded correctly
                        print(f"Processing file: {input_file_path}")
                        print(f"Data: {data}")
                        start_time = time.time()

                        messages = data.get('messages', [])
                        filled_data = label_emotion(data, messages)
                        end_time = time.time()

                        print(f"Processing {filename} took {end_time - start_time} seconds")
                        print(f"emotion_labeled_data_{filename}:", filled_data)
                        print("=====================================")

                        # Extract the number from the original file name
                        match = re.search(r'_(\d{5})\.json$', filename)
                        if match:
                            file_number = match.group(1)
                        else:
                            raise ValueError(f"Filename {filename} does not match the expected pattern")

                        output_filename = f"{dataset}_{data_state}_{folder_name}_{file_number}.json"
                        output_file_path = os.path.join(output_directory_path, output_filename)
                        os.makedirs(output_directory_path, exist_ok=True)

                        with open(output_file_path, 'w', encoding='utf-8') as file:
                            json.dump(filled_data, file, ensure_ascii=False, indent=4)

                        break  # If successful, break out of the retry loop

                    except RateLimitError as e:
                        print(f"RateLimitError: Attempt {retry + 1} of {max_retries}. Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                else:
                    source_path = os.path.join(input_directory_path, filename)
                    destination_path = os.path.join(except_directory_path, filename)
                    shutil.copy(source_path, destination_path)
                    print(f"File {filename} copied to {except_directory_path} due to RateLimitError.")

                    continue
            
            except Exception as e:
                print(f"An error occurred while processing {filename}: {e}")
            


if __name__ == "__main__":
    all_files = [f for f in os.listdir(input_directory_path) if f.endswith(".json")]
    all_files.sort()

    files_to_process = sorted(all_files[433:])
    
    process_files(files_to_process, input_directory_path, output_directory_path)