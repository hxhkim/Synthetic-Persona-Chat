import openai
import json
import os
from dotenv import load_dotenv
from typing import Dict, List
import time

load_dotenv()

MODEL = "gpt-4o-mini"
openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai



# New-SPC_2
folder_name = "New-SPC_2"
input_directory_path = f"/home/user1/conversation-data/dataset-02-SPC/Synthetic-Persona-Chat/data/02_renamed_data/{folder_name}"
output_directory_path = f"/home/user1/conversation-data/dataset-02-SPC/Synthetic-Persona-Chat/data/03_filled_data/{folder_name}"

# # test
# folder_name = "test"
# input_directory_path = f"/home/user1/conversation-data/dataset-02-SPC/Synthetic-Persona-Chat/data/02_renamed_data/{folder_name}"
# output_directory_path = f"/home/user1/conversation-data/dataset-02-SPC/Synthetic-Persona-Chat/data/03_filled_data/{folder_name}"



def fill_empty_fields(data: Dict, messages: List[Dict]) -> Dict:
    messages_str = "\n".join([f"{msg['role']}: {msg['content']['text']}" for msg in messages])
    
    participant_1 = data['participant_persona']['participant_1']
    participant_2 = data['participant_persona']['participant_2']

    prompt = f"""Analyze the given Conversation Log and Participant information and fill in ONLY the missing fields in the original JSON format. Do not modify any existing information.:

    Participant 1:
    name: {participant_1['name']}
    age: {participant_1['age']}
    gender: {participant_1['gender']}
    personality: {participant_1['personality']}
    background: {participant_1['background']}

    Participant 2:
    name: {participant_2['name']}
    age: {participant_2['age']}
    gender: {participant_2['gender']}
    personality: {participant_2['personality']}
    background: {participant_2['background']}

    # Conversation Log:
    {messages_str}

    # Guidelines:
    1. Infer age, gender, and other details based on the text content and writing style.
    2. Generate diverse and unique names and personalities for each participant. Use various expressions, not using the same expressions repeatedly.
    3. Use str sentences for the personality and background fields.
    4. Keep use the original fields text if it exists."""

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
                return

            for key, value in filled_data.items():
                if key == 'Participant 1':
                    participant_1.update(value)
                elif key == 'Participant 2':
                    participant_2.update(value)
            
            data['participant_persona']['participant_1'] = participant_1
            data['participant_persona']['participant_2'] = participant_2

            return data
        
        except openai.error.Timeout as e:
            print(f"Attempt {attempt + 1} of {retries} failed with timeout. Retrying...")
            time.sleep(3)

    raise Exception("All retry attempts failed due to timeout.")



def process_files(file_list: List[str], input_directory_path: str, output_directory_path: str):
    for filename in file_list:
        input_file_path = os.path.join(input_directory_path, filename)
        with open(input_file_path, 'r') as file:
            data = json.load(file)
        
        messages = data.get('messages', [])
        filled_data = fill_empty_fields(data, messages)
        
        if filled_data:
            file_name, _ = os.path.splitext(filename)
            file_number = file_name.split('_')[-1]
            output_filename = f"SPC-filled_data_{folder_name}_{file_number}.json"
            output_file_path = os.path.join(output_directory_path, output_filename)
            with open(output_file_path, 'w') as file:
                json.dump(filled_data, file, indent=4)




def count_empty_fields(output_directory_path):
    # Initialize counters
    counters = {
        "participant_1": {"name": 0, "age": 0, "gender": 0, "personality": 0, "background": 0},
        "participant_2": {"name": 0, "age": 0, "gender": 0, "personality": 0, "background": 0}
    }

    # List to store filenames with empty fields
    files_with_empty_fields = []

    # Iterate through each JSON file in the directory
    for filename in os.listdir(output_directory_path):
        if filename.endswith(".json"):
            file_path = os.path.join(output_directory_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                participant_persona = data.get("participant_persona", {})
                
                for participant in ["participant_1", "participant_2"]:
                    for field in ["name", "age", "gender", "personality", "background"]:
                        if not participant_persona.get(participant, {}).get(field):
                            counters[participant][field] += 1
                            if filename not in files_with_empty_fields:
                                files_with_empty_fields.append(filename)

    files_with_empty_fields = sorted(files_with_empty_fields)

    return files_with_empty_fields




def re_process_files(file_list: List[str], output_directory_path: str):
    for filename in file_list:
        input_file_path = os.path.join(output_directory_path, filename)
        with open(input_file_path, 'r') as file:
            data = json.load(file)
        
        messages = data.get('messages', [])
        filled_data = fill_empty_fields(data, messages)
        
        if filled_data:
            output_file_path = os.path.join(output_directory_path, filename)
            with open(output_file_path, 'w') as file:
                json.dump(filled_data, file, indent=4)


if __name__ == "__main__":
    # # Step 1: Run process_files once
    all_files = [f for f in os.listdir(input_directory_path) if f.endswith(".json")]
    all_files.sort()
    process_files(all_files, input_directory_path, output_directory_path)
    
    # Step 2: Count empty fields and process files in a loop
    while True:
        files_with_empty_fields = count_empty_fields(output_directory_path)
        if not files_with_empty_fields:
            print("All fields are filled.")
            break
        else:
            print(f"Processing {len(files_with_empty_fields)} files with empty fields...")
            re_process_files(files_with_empty_fields, output_directory_path)
            print("Processing complete. Checking for remaining empty fields...")
