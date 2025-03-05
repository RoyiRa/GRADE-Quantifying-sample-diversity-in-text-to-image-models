import pandas as pd
from openai import OpenAI
import json
import os
import math 
import base64
from utils import load_paths_for_prompt_id, load_oai_key

client = OpenAI(api_key=load_oai_key())    


def extract_attribute_values(concepts_dataset_path, vqa_model, generated_images_path, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    add_images_to_csv(concepts_dataset_path, generated_images_path)

 
    df = pd.read_csv(concepts_dataset_path.replace(".csv", "_with_images.csv"))
    
    # add None of the above to the attribute values
    df["attribute_values"] = df['attribute_values'].apply(lambda x: eval(x.strip().lower()))
    df['attribute_values'] = df['attribute_values'].apply(lambda x: set(list(x) + ["None of the above"]))
    
    new_rows = []
    for _, row in df.iterrows():
        question = row["attribute"]
        attribute_values = row["attribute_values"]
        encoded_img = encode_image(row["local_imagepath"]) # to work in scale, consider uploading to S3 and using URL instead
        encoded_img = f"data:image/jpeg;base64,{encoded_img}"
        reasoning_steps, answer = single_call(vqa_model, question, attribute_values, encoded_img)
        row["reasoning_steps"] = reasoning_steps
        row["answer"] = answer
        new_rows.append(row)


    new_df = pd.DataFrame(new_rows)
    new_df.to_csv(os.path.join(results_dir, "extracted_attribute_values.csv"), index=False)


def add_images_to_csv(concepts_dataset_path, generated_images_path):
    df = pd.read_csv(concepts_dataset_path)
    new_rows = []
    # Iterate over each row in the original DataFrame
    for _, row in df.iterrows():
        prompt_id = row['prompt_id']
        img_dirpath = os.path.join(generated_images_path, str(prompt_id))

        imagepaths = load_paths_for_prompt_id(img_dirpath)
 
        if not imagepaths:
            print(f"Did not find images for prompt_id {prompt_id}")
            continue
        # For each local_imagepath, create a new row in the new DataFrame
        for imagepath in imagepaths:
            new_row = row.to_dict()
            new_row['local_imagepath'] = imagepath
            new_rows.append(new_row)
    # Create a new DataFrame from the new rows
    new_df = pd.DataFrame(new_rows)    
    # Useful if you choose to use batch-inference
    new_df['image_name'] = new_df['local_imagepath'].apply(lambda x: x.split('/')[-1].replace(".jpeg", ""))



    new_df['custom_id'] = new_df.apply(lambda x: f"{x['concept_id']}_{x['attribute_id']}_{x['prompt_id']}_{x['image_name']}", axis=1)
    new_df.drop(columns=['image_name'], inplace=True)

    new_df.to_csv(concepts_dataset_path.replace(".csv", "_with_images.csv"), index=False)

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
  
def single_call(model, question, attribute_values, image_url):
    response = client.chat.completions.create(
        model=model,
        messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": [
                    {"type": "text", "text": f"Answer the following question with one of the categories. To come up with the correct answer, carefully analyze the image and think step-by-step before providing the final answer.\nQuestion: {question}\nCategories:{attribute_values}\nSelection:"},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]}
            ],
        response_format={
                        "type": "json_schema",
                        "json_schema": {
                        "name": "reasoning_schema",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                            "reasoning_steps": {
                                "type": "array",
                                "items": {
                                "type": "string"
                                },
                                "description": "The reasoning steps leading to the final conclusion."
                            },
                            "answer": {
                                "type": "string",
                                "description": "The final answer, taking into account the reasoning steps."
                            }
                            },
                            "required": ["reasoning_steps", "answer"],
                            "additionalProperties": False
                        }
                        }
                    },

        temperature=0.0,
        max_tokens=1000,

    )

    content = response.choices[0].message.content
    content = json.loads(content)
    print(content['reasoning_steps'])
    print(content['answer'])
    return content['reasoning_steps'], content['answer']
