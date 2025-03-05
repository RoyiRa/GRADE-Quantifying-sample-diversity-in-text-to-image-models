import argparse
import os
import sys
from data_generation import generate_data
from image_generation import generate_images
from vqa import extract_attribute_values
from distribution_estimation import compute_GRADEScore, report_default_behaviors

def load_concepts(concepts_path):
    """
    Load concepts from a file. Each line in the file represents a concept.
    """
    try:
        with open(concepts_path, 'r') as file:
            concepts = [line.strip() for line in file if line.strip()]
        return concepts
    except FileNotFoundError:
        print(f"Error: The file '{concepts_path}' does not exist.")
        exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Generate images based on provided concepts using specified models."
    )
    
    parser.add_argument(
        "--model_name",
        choices=[
            'sdxl', 'sdxl-turbo', 'sd-1.4', 'sd-2.1', 'lcm-sdxl',
            'deepfloyd-xl', 'deepfloyd-l', 'deepfloyd-m', 'sd-3',
            'sd-1.1', 'flux-schnell', 'flux-dev', 'google-search'
        ],
        default='sdxl',
        help="Name of the model you want to assess."
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="GPU ID to use for generating images."
    )
    parser.add_argument(
        "--concepts_path",
        type=str,
        default=None,
        help="Path to a file containing concepts (one per line). If provided, data will be generated based on these concepts."
    )
    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=100,
        help="Number of images to generate per concept."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for image generation."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Name of the dataset to be used/stored. Required if --concepts_path is not provided."
    )

    parser.add_argument(
        '--vqa_model',
        type=str,
        default="gpt-4o",
        help="Name of the model to use for VQA."
    )
    parser.add_argument(
        "--compute_for_single_prompt_distributions",
        type=bool,
        default=True,
        help="Compute diversity over a single prompt or multiple prompts."
    )

    parser.add_argument(
        "--report_default_behaviors",
        type=bool,
        default=False,
        help="Report the default behaviors for the measured concepts."
    )
    
    args = parser.parse_args()

    # # Step 1: generate prompts, attributes, and attribute values
    if not args.concepts_path and not args.dataset_name:
        parser.error("At least one of --concepts_path or --dataset_name must be provided.")

    dataset_dir = "datasets"
    if args.concepts_path:
        concepts = load_concepts(args.concepts_path)
        generate_data(concepts)
        dataset_name = "concepts_dataset"
        dataset_path = os.path.join("datasets", f"{dataset_name}.csv")
    else:
        if not args.dataset_name:
            parser.error("--dataset_name is required when --concepts_path is not provided.")
        dataset_name = args.dataset_name
        dataset_path = os.path.join(dataset_dir, f"{args.dataset_name}.csv")
        
        
        # Check if the dataset exists
        if not os.path.isfile(dataset_path):
            print(f"Error: The dataset file '{dataset_path}' does not exist.")
            sys.exit(1)
        else:
            print(f"Using existing dataset at '{dataset_path}'.")

    # Step 2: Generate images based on the determined dataset_path
    generate_images(
        model_name=args.model_name,
        dataset_path=dataset_path,
        num_images_to_generate=args.num_images_per_prompt,
        device=args.gpu_id,
        batch_size=args.batch_size
    )

    # Step 3: Extract attribute values from the generated images
    results_dir = os.path.join("results", args.model_name, dataset_name)
    generated_images_path = os.path.join("generated_images", args.model_name, dataset_name)
    extract_attribute_values(dataset_path, args.vqa_model, generated_images_path, results_dir)

    # Step 4: Assess diversity for multi-prompt distributions
    extracted_attribute_values_path = os.path.join(results_dir, 'extracted_attribute_values.csv')
    gradescore = compute_GRADEScore(extracted_attribute_values_path, is_single_prompt_dist=args.compute_for_single_prompt_distributions) # change is_single_prompt_dist to True for single-prompt distributions
    print(f"The GRADEScore of {args.model_name} is {gradescore}")

    # Step 5 (optional): Report default behaviors
    if args.report_default_behaviors:
        report_default_behaviors(os.path.join(results_dir, 'extracted_attribute_values.csv'), results_dir, is_single_prompt_dist=args.compute_for_single_prompt_distributions)
    
if __name__ == "__main__":
    main()
