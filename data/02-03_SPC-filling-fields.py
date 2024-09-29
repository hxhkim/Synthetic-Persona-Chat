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
client

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
            time.sleep(3)  # Wait for 3 seconds before retrying

    raise Exception("All retry attempts failed due to timeout.")



# # SPC-test
# # emotion labeling을 얘만 먼저 돌려버려서, 얘만 경로가 반대로 되어있음.
# # 이름을 바꾸는 추가 작업이 필요함.
# # 완료 (240929)

# json_files = sorted(glob.glob('/home/user1/conversation-data/dataset-02-SPC/Synthetic-Persona-Chat/data/04_emotion-labeled_data/SPC-test/*.json'))

# for i, json_file in enumerate(json_files):
#     with open(json_file, 'r') as file:
#         data = json.load(file)
#         messages = data['messages']

#     filled_data = fill_empty_fields(data, data['messages'])
#     print(f"filled_data_SPC-test_{i + 1}:", filled_data)
#     print(f"Function end time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
#     print("=====================================")

#     output_dir = '/home/user1/conversation-data/dataset-02-SPC/Synthetic-Persona-Chat/data/03_filled_data/SPC-test'
#     output_file = os.path.join(output_dir, f'filled_data_SPC-test_{i + 1}.json')

#     with open(output_file, 'w', encoding='utf-8') as file:
#         json.dump(filled_data, file, ensure_ascii=False, indent=4)

# # SPC-train

# json_files = sorted(glob.glob('/home/user1/conversation-data/dataset-02-SPC/Synthetic-Persona-Chat/data/02_renamed_data/SPC-train/*.json'))

# for i, json_file in enumerate(json_files[:5]):
#     with open(json_file, 'r') as file:
#         data = json.load(file)
#         messages = data['messages']

#     filled_data = fill_empty_fields(data, data['messages'])
#     print(f"filled_data_SPC-train_{i + 1}:", filled_data)
#     print(f"Function end time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
#     print("=====================================")

#     output_dir = '//home/user1/conversation-data/dataset-02-SPC/Synthetic-Persona-Chat/data/03_filled_data/SPC-train'
#     output_file = os.path.join(output_dir, f'filled_data_SPC-train_{i + 1}.json')

#     with open(output_file, 'w', encoding='utf-8') as file:
#         json.dump(filled_data, file, ensure_ascii=False, indent=4)

# SPC-valid

json_files = sorted(glob.glob('/home/user1/conversation-data/dataset-02-SPC/Synthetic-Persona-Chat/data/02_renamed_data/SPC-valid/*.json'))

for i, json_file in enumerate(json_files):
    with open(json_file, 'r') as file:
        data = json.load(file)
        messages = data['messages']

    filled_data = fill_empty_fields(data, data['messages'])
    print(f"filled_data_SPC-valid_{i + 1}:", filled_data)
    print(f"Function end time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    print("=====================================")

    output_dir = '//home/user1/conversation-data/dataset-02-SPC/Synthetic-Persona-Chat/data/03_filled_data/SPC-valid'
    output_file = os.path.join(output_dir, f'filled_data_SPC-valid_{i + 1}.json')

    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(filled_data, file, ensure_ascii=False, indent=4)


# # New-SPC

# json_files = sorted(glob.glob('/home/user1/conversation-data/dataset-02-SPC/Synthetic-Persona-Chat/data/02_renamed_data/New-SPC/*.json'))

# for i, json_file in enumerate(json_files):
#     with open(json_file, 'r') as file:
#         data = json.load(file)
#         messages = data['messages']

#     filled_data = fill_empty_fields(data, data['messages'])
#     print(f"filled_data_New-SPC_{i + 1}:", filled_data)
#     print(f"Function end time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
#     print("=====================================")

#     output_dir = '//home/user1/conversation-data/dataset-02-SPC/Synthetic-Persona-Chat/data/03_filled_data/New-SPC'
#     output_file = os.path.join(output_dir, f'filled_data_New-SPC_{i + 1}.json')

#     with open(output_file, 'w', encoding='utf-8') as file:
#         json.dump(filled_data, file, ensure_ascii=False, indent=4)