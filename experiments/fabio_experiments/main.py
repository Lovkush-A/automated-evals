import os
import argparse
from dotenv import load_dotenv
from openai import OpenAI
from idea_generation import generate_evaluation_idea, save_idea_to_json
from dataset_generation import main as generate_dataset
from scoring import main as score_dataset

# Update model definitions
SCORER_MODEL = "gpt-4-turbo-2024-04-09"  # Previously might have been a different model
QUESTIONS_MODEL = "gpt-4-turbo-2024-04-09"  # For generating questions
IDEA_MODEL = "gpt-3.5-turbo-1106"  # New model for idea generation

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
    
    # Parser for the 'score' command
    score_parser = subparsers.add_parser("score", help="Score a dataset")
    score_parser.add_argument("dataset", help="Path to the dataset JSON file")
    score_parser.add_argument("--output-dir", help="Directory to save the scores and visualizations")
    score_parser.add_argument("--no-analyze", action="store_true", help="Skip analysis and visualization")
    
    # Common arguments
    parser.add_argument("--api-key", type=str, help="OpenAI API key (if not set in environment)")
    
    args = parser.parse_args()
    
    
    
    # Set API key if provided
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key
    
    load_dotenv()
    
    # Access the API key from the environment
    api_key = os.getenv("OPENAI_API_KEY")

    # Use the API key to initialize the OpenAI client
    if api_key:
        client = OpenAI(api_key=api_key)
    else:
        print("Error: OpenAI API key not found. Please set the OPENAI_API_KEY environment variable or use --api-key.")
        return
    
    # Initialize OpenAI client
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    # Execute the appropriate command
    if args.command == "generate":
        if args.generate_idea_only:
            # Only generate and save the evaluation idea
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
