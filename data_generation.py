from openai import OpenAI
from utils import load_oai_key
import pandas as pd
import os
from typing import List

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
        Do NOT ask questions pertaining to relative attributes (e.g., "Is the cake big or small?").

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
                'attribute': question
            })
        seen_ids.add(c_id)
    concepts_questions_df = pd.DataFrame(concepts_questions)
    # add question_id column
    concepts_questions_df['attribute_id'] = concepts_questions_df.index
    concepts_questions_df.to_csv("datasets/concepts_questions.csv", index=False)
     
  
def generate_attribute_values():
    prompts_df = pd.read_csv("datasets/concepts_prompts.csv")
    questions_df = pd.read_csv("datasets/concepts_questions.csv")
    df = prompts_df.merge(questions_df, on=['concept', 'concept_id'])


    captions = df['prompt'].tolist()
    questions = df['attribute'].tolist()
    caption_ids = df['prompt_id'].tolist()
    question_ids = df['attribute_id'].tolist()
    concept_ids = df['concept_id'].tolist()
    concepts = df['concept'].tolist()

    concepts_attribute_values = []
    seen_ids = set()
    for caption, question, caption_id, q_id, concept, c_id in zip(captions, questions, caption_ids, question_ids, concepts, concept_ids):
        curr_id = f"{caption_id}_{q_id}"
        if curr_id in seen_ids:
            continue
        instruction = f"""
                        I have a question that is asked about an image. I will provide you with the question and a caption of the image. Your job is to first carefully read the question and analyze, then hypothesize plausible answers to the question assuming you could examine the image (instead, you examine the caption). 
                        The answers should be in a list, as in the example below. 
                        Do not write anything other than the plausible answers.
                        Do not provide extra details to your answers in parentheses (e.g., white and NOT 'white (for decorated cookies)').
                        Do your best to be succinct and not overly-specific.
                        If the question is very open-ended, like 'Is there anything on the table?' or 'Is the cake decorated with any specific theme or design?', the answers should be strictly 'yes' or 'no'.
                        
                        Example:
                        Caption: a helmet in a bike shop
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
                        Caption: {caption}
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
        seen_ids.add(curr_id)

        # add attribute_values
        concepts_attribute_values.append({
            'concept_id': c_id,
            'concept': concept,
            'prompt_id': caption_id,
            'prompt': caption,
            'attribute_id': q_id,
            'attribute': question,
            'attribute_values': attribute_values
        })

    concepts_attribute_values_df = pd.DataFrame(concepts_attribute_values)
    concepts_attribute_values_df.to_csv("datasets/unfiltered_concepts_dataset.csv", index=False)



def filter_duplicate_attribute_values():
    df = pd.read_csv("datasets/unfiltered_concepts_dataset.csv")
    df['attribute_values'] = df['attribute_values'].apply(eval)

    attribute_values_dict = {}  # Initialize an empty dictionary to store results

    for (concept, question), group in df.groupby(['concept', 'attribute']):
        attribute_values = group['attribute_values'].tolist()        
        attribute_values = [item.lower() for sublist in attribute_values for item in sublist]
        attribute_values = list(set(attribute_values))


        attribute_values_dict[(concept, question)] = attribute_values
    
    filtered_attribute_values = {}
    for key, attribute_values in attribute_values_dict.items():
        concept, question = key

        instruction = f"""
        You are provided with a concept, a question that is asked about an image, and possible answers to the question. 
        Your job is to filter out answers. There are three criteria that lead to deleting an answer: (1) semantic redundancy--when there is already a similar answer in the list; (2) difficulty to detect from viewing an image--an answer that cannot be answered conclusively by viewing an image; (3) out of scope--an answer that is different in category from the rest of the answers; (4) None of the above--an answer that essentially indicates no answer is right.        
        Start by first carefully reading the question and possible answers, then determine which answers are redundant. 
        For semantic redundancies, opt to remove the more specific answer (e.g., remove 'chocolate drizzle' instead of 'chocolate').
        EVERY answer should be in a list format e.g., ['answer1', 'answer2', 'answer3'].
        Do NOT write ANYTHING except for the filtered answers.
        Do NOT allow for answers like None of the above, e.g., 'none' or 'no visible toppings or additions'.
        
        Example 1:
        Concept: popcorn
        Question: Are there any visible toppings or additions, such as butter or cheese?
        Answers: ['no', 'yes', 'salt', 'chocolate', 'cinnamon', 'butter', 'none', 'chocolate drizzle', 'no visible toppings or additions', 'plain', 'seasoning', 'herbs', 'truffle oil', 'caramel', 'spices', 'sugar', 'cheese']
        Explanation: 'none' matches criterion (4), 'chocolate' and 'chocolate drizzle' are similar and match criterion (1). We remove 'chocolate drizzle', as it is more specific. 'no visible toppings or additions' matches (4). 'seasoning', 'herbs', 'spices', 'salt', 'sugar', and 'truffle oil' match criterion (2), since they're too difficult to identify from an image of popcorn. 'no' and 'yes' are out of scope (criterion 3), since all answers describe specific toppings while these answers are general.
        Filtered Answers: ['chocolate', 'cinnamon', 'butter', 'plain', 'caramel', 'cheese']

        Example 2:
        Concept: a table
        Question: How many legs does the table have?
        Answers: Attribute values: ['no legs', 'no', 'yes', 'one central pedestal', 'one leg', 'two trestle supports', 'a trestle base', 'two legs', 'six legs', 'a pedestal base', 'three legs', 'multiple legs', 'five legs', 'four legs']
        Explanation: 'no leg' matches criterion (4), anything with 'trestle' is too specific and out of scope (criterion 3), 'two legs', 'three legs', 'four legs', 'five legs', 'six legs' are semantically redundant (criterion 1) and are more specific compared to 'multiple legs'.
        Filtered Answers: ['one leg', 'multiple legs']

        Now your turn.

        Concept: {concept}
        Question: {question}
        Answers: {attribute_values}
        Filtered answers:
        """


        response = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "user", "content": [{"type": "text", "text": instruction}]}
                    ],
                    temperature=0.0
                    )
                
        content = response.choices[0].message.content.strip().lower()
        filtered_list = eval(content)
        if filtered_list:
            filtered_attribute_values[key] = filtered_list

            print(f"Question: {question}")
            print(f"Attribute values: {attribute_values}")
            print(f"Filtered attribute values: {filtered_attribute_values[key]}")
            print()



    # Create DataFrame from filtered_attribute_values
    filtered_df = pd.DataFrame({
        'concept': [concept for (concept, _) in filtered_attribute_values.keys()],
        'attribute': [question for (_, question) in filtered_attribute_values.keys()],
        'attribute_values': [filtered_attribute_values[(concept, question)] for (concept, question) in filtered_attribute_values.keys()]
    })

    # Get unique rows from df with the additional columns
    df_unique = df[['concept', 'attribute', 'concept_id', 'attribute_id', 'prompt', 'prompt_id']].drop_duplicates()

    # Merge filtered_df with df_unique on ['concept', 'question']
    final_df = filtered_df.merge(df_unique, on=['concept', 'attribute'], how='left')

    # Reorder columns
    final_df = final_df[['concept', 'attribute', 'concept_id', 'attribute_id', 'prompt', 'prompt_id', 'attribute_values']]

    
    # Save to CSV without index
    final_df.to_csv("datasets/concepts_dataset.csv", index=False)

    
def generate_data(concepts: List[str]):
    print(f'Starting...')
    generate_prompts(concepts)
    generate_attributes()
    generate_attribute_values()
    filter_duplicate_attribute_values()