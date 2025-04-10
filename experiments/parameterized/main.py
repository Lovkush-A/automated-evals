import os
import argparse
import json
from dotenv import load_dotenv
from openai import OpenAI
from idea_generation import generate_evaluation_idea, generate_evaluation_idea_from_parameters, save_idea_to_json, load_json_file
from dataset_generation import main as generate_dataset
from scoring import main as score_dataset

# Update model definitions
SCORER_MODEL = "gpt-4-turbo-2024-04-09"  # Previously might have been a different model
QUESTIONS_MODEL = "gpt-4-turbo-2024-04-09"  # For generating questions
IDEA_MODEL = "gpt-3.5-turbo-1106"  # New model for idea generation

def list_available_parameters():
    """Display available parameters from the configuration files"""
    threat_models = load_json_file("experiments/parameterized/parameters/threat_models.json")
    elicitation_dimensions = load_json_file("experiments/parameterized/parameters/elicitation_dimensions.json")
    
    print("\nAvailable Threat Models:")
    for model_id, model_data in threat_models.get("threat_models", {}).items():
        print(f"- {model_id}: {model_data.get('name')}")
    
    print("\nAvailable Elicitation Dimensions:")
    for dim_type, dim_data in elicitation_dimensions.get("elicitation_dimensions", {}).items():
        print(f"\n{dim_type}:")
        print(f"  Description: {dim_data.get('description', '')}")
        print("  Values:")
        for value_id, value_data in dim_data.get("values", {}).items():
            print(f"  - {value_id}: {value_data.get('name', '')}")

def main():
    parser = argparse.ArgumentParser(description="Generate and evaluate AI evaluation datasets")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Parser for the 'generate' command
    generate_parser = subparsers.add_parser("generate", help="Generate a dataset")
    generate_parser.add_argument("--generate-idea-only", action="store_true", help="Only generate the evaluation idea, not the dataset")
    generate_parser.add_argument("--use-existing-idea", action="store_true", help="Use existing evaluation idea from file")
    generate_parser.add_argument("--num-questions", type=int, default=4, help="Number of questions to generate")
    generate_parser.add_argument("--no-concurrent", action="store_true", help="Disable concurrent generation")
    generate_parser.add_argument("--overwrite", action="store_true", help="Overwrite existing dataset.json file")
    generate_parser.add_argument("--full", action="store_true", 
                                help="Use iterative process to generate high-quality questions until convergence")
    
    # Add new parameterized arguments
    generate_parser.add_argument("--parameterized", action="store_true", 
                                help="Use parameterized generation instead of random")
    generate_parser.add_argument("--threat-model", type=str, 
                                help="Specific threat model to use (e.g., 'power_seeking')")
    generate_parser.add_argument("--dimension-types", type=str, nargs='+',
                                help="Types of elicitation dimensions (e.g., 'goal_types context_types')")
    generate_parser.add_argument("--dimension-values", type=str, nargs='+',
                                help="Dimension values corresponding to dimension types (must match order)")
    generate_parser.add_argument("--dimension-type", type=str, 
                                help="Type of elicitation dimension (legacy, use --dimension-types instead)")
    generate_parser.add_argument("--dimension-value", type=str, 
                                help="Specific dimension value (legacy, use --dimension-values instead)")
    generate_parser.add_argument("--list-parameters", action="store_true",
                                help="List available threat models and elicitation dimensions")
    
    # Parser for the 'score' command
    score_parser = subparsers.add_parser("score", help="Score a dataset")
    score_parser.add_argument("dataset", help="Path to the dataset JSON file")
    score_parser.add_argument("--output-dir", help="Directory to save the scores and visualizations")
    score_parser.add_argument("--no-analyze", action="store_true", help="Skip analysis and visualization")
    
    # Common arguments
    parser.add_argument("--api-key", type=str, help="OpenAI API key (if not set in environment)")
    
    args = parser.parse_args()
    
    # List available parameters if requested
    if args.command == "generate" and args.list_parameters:
        list_available_parameters()
        return
    
    # Load environment variables from .env file first
    print("Current working directory:", os.getcwd())
    
    # Check if the .env file exists
    if os.path.exists('.env'):
        load_dotenv()
        print(".env file found and loaded.")
    else:
        print(".env file not found.")
    
    # Override with command-line API key if provided
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key
        print("Using API key provided via command line.")
    
    # Access the API key from the environment
    api_key = os.getenv("OPENAI_API_KEY")
    
    # Debug the API key state without exposing it
    if api_key:
        masked_key = f"***{api_key[-4:]}" if len(api_key) > 4 else "***"
        print(f"API key found (ending with: {masked_key}).")
    else:
        print("API key not found in environment variables.")
    
    # Use the API key to initialize the OpenAI client
    if api_key and api_key.strip():
        client = OpenAI(api_key=api_key)
    else:
        print("Error: OpenAI API key not found or is empty. Please set the OPENAI_API_KEY environment variable or use --api-key.")
        return
    
    # Execute the appropriate command
    if args.command == "generate":
        # Check if we need to use parameterized generation
        if args.parameterized:
            # Handle both new multi-dimension and legacy single-dimension parameters
            has_multi = args.dimension_types is not None and args.dimension_values is not None
            has_single = args.dimension_type is not None and args.dimension_value is not None
            
            if not args.threat_model:
                print("Error: When using --parameterized, you must specify --threat-model")
                print("Use --list-parameters to see available options")
                return
            
            if not (has_multi or has_single):
                print("Error: When using --parameterized, you must specify either:")
                print("  1. --dimension-types and --dimension-values (for multiple dimensions)")
                print("  2. --dimension-type and --dimension-value (for single dimension, legacy)")
                print("Use --list-parameters to see available options")
                return
            
            if has_multi and has_single:
                print("Warning: Both new and legacy dimension parameters provided. Using the new multi-dimension format.")
            
            # Generate idea from parameters
            print(f"Generating parameterized evaluation idea...")
            print(f"Threat Model: {args.threat_model}")
            
            if has_multi:
                if len(args.dimension_types) != len(args.dimension_values):
                    print("Error: The number of dimension types must match the number of dimension values")
                    return
                    
                print("Dimensions:")
                for d_type, d_value in zip(args.dimension_types, args.dimension_values):
                    print(f"  - {d_type}: {d_value}")
                
                idea = generate_evaluation_idea_from_parameters(
                    client=client,
                    threat_model=args.threat_model,
                    dimension_types=args.dimension_types,
                    dimension_values=args.dimension_values
                )
            else:
                print(f"Dimension Type: {args.dimension_type}")
                print(f"Dimension Value: {args.dimension_value}")
                
                idea = generate_evaluation_idea_from_parameters(
                    client=client,
                    threat_model=args.threat_model,
                    dimension_type=args.dimension_type,
                    dimension_value=args.dimension_value
                )
            
            # Save the idea and continue based on other arguments
            print("\nGenerated evaluation idea:")
            for key, value in idea.items():
                if key != "metadata":
                    print(f"{key}: {value}")
            save_idea_to_json(idea)
            
            # If generate-idea-only, stop here
            if args.generate_idea_only:
                return
            
            # Otherwise, continue to dataset generation with this idea
            generate_dataset(
                use_existing_idea=True,  # We just saved the idea, so use it
                num_questions=args.num_questions,
                concurrent=not args.no_concurrent,
                overwrite=args.overwrite
            )
        elif args.generate_idea_only:
            # Only generate and save the evaluation idea (non-parameterized)
            print("Generating evaluation idea...")
            idea = generate_evaluation_idea(client)
            print("\nGenerated evaluation idea:")
            for key, value in idea.items():
                print(f"{key}: {value}")
            save_idea_to_json(idea)
        elif args.full:
            # Use the iterative process to generate high-quality questions
            from dataset_generation import iterate_dataset_generation
            print("Starting full iterative dataset generation process...")
            iterate_dataset_generation(
                client=client,
                num_questions=args.num_questions,
                overwrite=args.overwrite
            )
        else:
            # Generate the dataset using the standard approach
            generate_dataset(
                use_existing_idea=args.use_existing_idea, 
                num_questions=args.num_questions,
                concurrent=not args.no_concurrent,
                overwrite=args.overwrite
            )
    elif args.command == "score":
        # Score the dataset
        score_dataset(args.dataset, args.output_dir, not args.no_analyze)
    else:
        # If no command is specified, show help
        parser.print_help()

if __name__ == "__main__":
    main()
