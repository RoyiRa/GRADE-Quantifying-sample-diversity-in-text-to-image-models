import os
import pandas as pd
import math

def load_model_predictions(filename):
    df = pd.read_csv(filename)

    df["attribute_values"] = df['attribute_values'].apply(lambda x: eval(x.strip().lower()))
    df['answer'] = df['answer'].apply(lambda x: x.strip().lower())

    initial_len = len(df)
    df = df[df['answer'] != 'none of the above']
    df = df[df['answer'] != 'neither']
    final_len = len(df)
    print(f"Removed {initial_len - final_len} rows with 'None of the above' answers ({1 - round(final_len/initial_len, 2)}).")
    return df


def default_behavior_threshold(series):
    counts = series.value_counts(normalize=True)
    max_percent = counts.max()
    if max_percent >= 0.8:
        return True, counts.idxmax(), max_percent
    else:
        return False, counts.idxmax(), max_percent
    

def extract_default_behaviors(filename, results_dir, is_single_prompt_dist=False):
    df = load_model_predictions(filename)

    if is_single_prompt_dist:
        result = df.groupby(['prompt', 'question'])['answer'].apply(default_behavior_threshold).reset_index()
    else:
        result = df.groupby(['concept', 'question'])['answer'].apply(default_behavior_threshold).reset_index()
  
    result[['has_dominant', 'default_behavior', 'percentile']] = pd.DataFrame(result['answer'].tolist(), index=result.index)
    result.drop(columns=['answer'], inplace=True)


    destdir = os.path.join(results_dir, "default_behaviors")
    os.makedirs(destdir, exist_ok=True)
    if is_single_prompt_dist:
        result.to_csv(os.path.join(destdir, 'single_prompt_default_behaviors.csv'), index=False)
    else:
        result.to_csv(os.path.join(destdir, 'multi_prompt_default_behaviors.csv'), index=False)


def report_default_behaviors(filename, results_dir, is_single_prompt_dist=False):
    extract_default_behaviors(filename, results_dir, is_single_prompt_dist)
    if is_single_prompt_dist:
        df = pd.read_csv(os.path.join(results_dir, "default_behaviors", 'single_prompt_default_behaviors.csv'))
        unique_with_at_least_one_true = df[df['has_dominant'] == True]['prompt'].drop_duplicates()
        at_least_one_true_ratio = len(unique_with_at_least_one_true) / len(df['prompt'].drop_duplicates())
    else:
        df = pd.read_csv(os.path.join(results_dir, "default_behaviors", 'multi_prompt_default_behaviors.csv'))
        unique_with_at_least_one_true = df[df['has_dominant'] == True]['concept'].drop_duplicates()
        at_least_one_true_ratio = len(unique_with_at_least_one_true) / len(df['concept'].drop_duplicates())
    
    total_default_behavior_ratio = df['has_dominant'].sum() / len(df)

    if is_single_prompt_dist:
        print(f"###Single prompt####")
    else:
        print(f"###Multi prompt####")
    print(f"Unique with at least one True: {round(at_least_one_true_ratio * 100, 2)}%")
    print(f"Total Default Behavior ratio: {round(total_default_behavior_ratio * 100, 2)}%")



def create_distributions(filename, is_single_prompt_dist=False):
    df = load_model_predictions(filename)
    out_of_scope_answers = {}
  
    def count_answers(group, group_keys):
        if is_single_prompt_dist:
            prompt_id, question_id = group_keys
            key = f"{prompt_id}_{question_id}"
        else:
            concept_id, question_id = group_keys
            key = f"{concept_id}_{question_id}"

        answer_counts = {}
        attribute_values = group['attribute_values'].iloc[0] # Assume all rows in a group have the same set
        for answer in attribute_values:
            answer_counts[answer] = 0
            

        # Count each answer in the 'answer' column
        for answer in group['answer']:
            if answer in answer_counts:
                answer_counts[answer] += 1
    
            if is_single_prompt_dist:
                key = f"{group['prompt'].iloc[0]}_{group['question'].iloc[0]}"
            else:
                key = f"{group['concept'].iloc[0]}_{group['question'].iloc[0]}"
                
            if answer not in answer_counts:   
                if key in out_of_scope_answers:
                    out_of_scope_answers[key][1] += 1
                else:
                    out_of_scope_answers[key] = [answer, 1]

        total = sum(answer_counts.values())
        for key, count in answer_counts.items():
            answer_counts[key] = count / total

        return answer_counts
    
    # Choose the appropriate grouping columns
    if is_single_prompt_dist:
        grouping_columns = ['prompt', 'question']
    else:
        grouping_columns = ['concept', 'question']

    result = df.groupby(grouping_columns).apply(
        lambda group: count_answers(group, group.name)
    )

    distributions = {f"{idx[0]}_{idx[1]}": counts for idx, counts in result.items()}
    return distributions


def compute_diversity_score(filename, is_single_prompt_dist=False):
    distributions = create_distributions(filename, is_single_prompt_dist)
    normalized_entropies = []
    entropies_as_dict = {}
    
    for key, dist in distributions.items():
        probabilities = [count for count in dist.values()]
        entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)  # Avoid log(0) error

        num_answers = len(dist)
        if num_answers > 1:  # Avoid log(0) error when there's only one type of answer
            max_possible_entropy = math.log2(num_answers)
            normalized_entropy = entropy / max_possible_entropy
        else:
            normalized_entropy = 0
        normalized_entropies.append(normalized_entropy)
        entropies_as_dict[key] = normalized_entropy


    mean_normalized_entropy = sum(normalized_entropies) / len(normalized_entropies)
    print(f"Mean Normalized Entropy: {mean_normalized_entropy}")

    return distributions, entropies_as_dict



if __name__ == "__main__":
    pass
    model_name = 'flux-schnell'
    dataset_name = "concepts_dataset"
    results_dir = os.path.join("results", model_name, dataset_name)
    report_default_behaviors(os.path.join(results_dir, 'extracted_attribute_values.csv'), results_dir, is_single_prompt_dist=False)
    compute_diversity_score(os.path.join(results_dir, 'extracted_attribute_values.csv'), is_single_prompt_dist=False)