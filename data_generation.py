from openai import OpenAI
from utils import load_oai_key
import pandas as pd
import os


client = OpenAI(api_key=load_oai_key())    
MODEL = "gpt-4o"


def generate_prompts(concepts: set):
    def create_instruction(concept, typical=True):
        setting_type = "typical" if typical else "atypical"
        example_settings = (
            '"a cake in a bakery",\n'
            '"a cake at a birthday party",\n'
            '"a cake at a swimming pool"'
        ) if typical else (
            '"a cake on a weight loss clinic",\n'
            '"a cake at a gym",\n'
            '"a cake at a swimming pool"'
        )
        return f"""Please suggest three {setting_type} settings for the concept below.
                Note that the output should be a list of strings.
                Here's an example:
                Concept: a cake
                Prompts: [
                {example_settings}
                ]
                Concept: {concept}"""

    def get_settings(prompt):
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "user", "content": [{"type": "text", "text": prompt}]}
            ],
            temperature=0.0
        )
        content = response.choices[0].message.content
        try:
            settings_str = content.split('[', 1)[1].rsplit(']', 1)[0]
            settings = eval(f"[{settings_str}]")
            return settings
        except (IndexError, SyntaxError, NameError) as e:
            print(f"Error parsing settings: {e}")
            return []

    
    prompts_data = []
    concept_to_id = {concept: idx for idx, concept in enumerate(concepts)}
    for concept in concepts:
        concept_id = concept_to_id[concept]
        common_instruction = create_instruction(concept, typical=True)
        common_prompts = get_settings(common_instruction)
        uncommon_instruction = create_instruction(concept, typical=False)
        uncommon_prompts = get_settings(uncommon_instruction)
        
        # Add typical prompts to the data list
        for prompt in common_prompts:
            prompts_data.append({
                'prompt': prompt,
                'concept': concept,
                'concept_id': concept_id
            })
        
        # Add atypical prompts to the data list
        for prompt in uncommon_prompts:
            prompts_data.append({
                'prompt': prompt,
                'concept': concept,
                'concept_id': concept_id
            })
    
    df = pd.DataFrame(prompts_data, columns=['prompt', 'concept', 'concept_id'])    
    df = df.reset_index().rename(columns={'index': 'prompt_id'})
    os.makedirs('datasets', exist_ok=True)
    df.to_csv(f'datasets/concepts_prompts.csv', index=False)

def generate_attributes():
    df = pd.read_csv("datasets/concepts_prompts.csv")
    concepts = df['concept'].tolist()
    concept_ids = df['concept_id'].tolist()
    # List to store new rows with concepts and their questions
    concepts_questions = []

    seen_ids = set()
    
    for c_id, concept in zip(concept_ids, concepts):
        if c_id in seen_ids:
            continue
        instruction = f"""
        Help me ask questions about images that depict certain concepts.
        I will provide you a concept. Your job is to analyze the concept's typical attributes and ask simple questions that can be answered by viewing the image. Your questions should involve concrete attributes.
        Do not ask more than 5 questions.

        Here's an example:
        concept: a cake
        attributes: cakes can be made in different flavors, shapes, and can have multiple tiers.
        questions:
        1. Is the cake eaten?
        2. Does the cake have multiple tiers?
        3. In what flavor is the cake?
        4. What is the shape of the cake?
        5. Does the cake show any signs of fruit on the outside or
        suggest a fruit flavor?

        Now that you understand, let's begin.
        concept: {concept}
        """
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "user", "content": [{"type": "text", "text": instruction}]}
            ],
            temperature=0.0
            )
        
        content = response.choices[0].message.content
        lines = content.split('\n')
        questions = [line[line.index(' ') + 1:] for line in lines if line.startswith(tuple(str(i) + '.' for i in range(1, 10)))]
        for question in questions:
            concepts_questions.append({
                'concept_id': c_id,
                'concept': concept,
                'question': question
            })
        seen_ids.add(c_id)
    concepts_questions_df = pd.DataFrame(concepts_questions)
    # add question_id column
    concepts_questions_df['question_id'] = concepts_questions_df.index
    concepts_questions_df.to_csv("datasets/concepts_questions.csv", index=False)
     
  
def generate_attribute_values():
    df = pd.read_csv("datasets/concepts_questions.csv")
    concepts = df['concept'].tolist()
    questions = df['question'].tolist()
    concept_ids = df['concept_id'].tolist()
    question_ids = df['question_id'].tolist()

    concepts_attribute_values = []
    seen_ids = set()
    for concept, question, c_id, q_id in zip(concepts, questions, concept_ids, question_ids):
        curr_id = f"{c_id}_{q_id}"
        if curr_id in seen_ids:
            continue
        instruction = f"""
                    I have a question that is asked about an image.  I will provide you with the question and a caption of the image. 
                    Your job is to analyze the description of the image and the question, hypothesize plausible answers that can surface from viewing the image. Do not write anything other than the answer.
                    Then, I need you to list the plausible answers in a list, just like in the example below. For example,
                    Caption: a helmet in a bike
                    Question: What type of helmet is depicted in the image? 
                    Plausible answers: ["motorcycle helmets",
                                "bicycle helmets",
                                "football helmets",
                                "construction helmets",
                                "military helmets",
                                "firefighter helmets",
                                "rock climbing helmets",
                                "hockey helmets"]
                    Now your turn. 
                    Caption: {concept}
                    Question: {question} 
                    Plausible answers:
                    """
        response = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "user", "content": [{"type": "text", "text": instruction}]}
                    ],
                    temperature=0.0
                    )
                
        attribute_values = response.choices[0].message.content
        print(attribute_values)
        print(type(attribute_values))

        seen_ids.add(curr_id)

        # add attribute_values
        concepts_attribute_values.append({
            'concept_id': c_id,
            'concept': concept,
            'question_id': q_id,
            'question': question,
            'attribute_values': attribute_values
        })

    concepts_attribute_values_df = pd.DataFrame(concepts_attribute_values)
    concepts_attribute_values_df.to_csv("datasets/concepts_attribute_values.csv", index=False)


if __name__ == "__main__":
    # concepts = ["a cookie", "an umbrella", "popcorn", "soap"]
    # generate_prompts(concepts)

    # generate_attributes()
    # generate_attribute_values()

    pass



  
