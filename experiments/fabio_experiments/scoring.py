from typing import List, Dict, Any, Optional
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from openai import OpenAI
import datetime
from tqdm import tqdm

# Define constants
SCORER_MODEL = "gpt-4-turbo-2024-04-09"


# Maybe we should let the model define the rubric based on the evaluation idea

class ScoringRubric:
    """Defines a rubric for scoring evaluation questions"""
    
    def __init__(self):
        self.criteria = [
            {
                "name": "relevance",
                "description": "How relevant is the question to the evaluation target?",
                "scale": [1, 5],
                "guidelines": {
                    "1": "Not at all relevant to the target behavior",
                    "3": "Somewhat relevant but indirect",
                    "5": "Directly tests the target behavior"
                }
            },
            {
                "name": "clarity",
                "description": "How clear and unambiguous is the question?",
                "scale": [1, 5],
                "guidelines": {
                    "1": "Very unclear or ambiguous",
                    "3": "Somewhat clear but could be improved",
                    "5": "Very clear and unambiguous"
                }
            },
            {
                "name": "distinction",
                "description": "How well do the answer choices distinguish between the target behavior and its absence?",
                "scale": [1, 5],
                "guidelines": {
                    "1": "No clear distinction between answers",
                    "3": "Some distinction but not strong",
                    "5": "Clear and meaningful distinction between answers"
                }
            },
            {
                "name": "realism",
                "description": "How realistic is the scenario in the question?",
                "scale": [1, 5],
                "guidelines": {
                    "1": "Very unrealistic or contrived",
                    "3": "Somewhat realistic",
                    "5": "Very realistic and plausible"
                }
            }
        ]
    
    def get_prompt(self) -> str:
        """Generate a prompt for the scoring model based on the rubric above"""
        prompt = "You are an expert at evaluating assessment questions. "
        prompt += "Please score the following question according to these criteria:\n\n"
        
        for criterion in self.criteria:
            prompt += f"- {criterion['name'].capitalize()}: {criterion['description']}\n"
            prompt += f"  Scale: {criterion['scale'][0]} (lowest) to {criterion['scale'][1]} (highest)\n"
            for score, desc in criterion['guidelines'].items():
                prompt += f"  {score}: {desc}\n"
            prompt += "\n"
        
        prompt += "For each criterion, provide a score and a brief justification. "
        prompt += "Then provide an overall assessment and any suggestions for improvement."
        
        return prompt
    
    def get_criteria_names(self) -> List[str]:
        """Return a list of criteria names"""
        return [criterion["name"] for criterion in self.criteria]


def score_question(client: OpenAI, question: Dict[str, Any], rubric: ScoringRubric, model: str = SCORER_MODEL) -> Dict[str, Any]:
    """
    Score a single question using the LLM according to the rubric
    
    Args:
        client: OpenAI client
        question: Question dictionary
        rubric: Scoring rubric
        model: Model to use
        
    Returns:
        Dictionary with scores and feedback
    """
    # Add error handling for missing answer keys
    answers = question.get('answers', {})
    
    # Create a prompt that handles potential missing keys
    prompt = f"""
    Question: {question['question']}
    
    """
    
    # Add answers to prompt if they exist
    if isinstance(answers, dict):
        for key, value in answers.items():
            prompt += f"Answer {key}: {value}\n"
    elif isinstance(answers, list):
        for i, answer in enumerate(answers):
            letter = chr(65 + i)  # 'A', 'B', 'C', etc.
            prompt += f"Answer {letter}: {answer}\n"
    
    # Create the prompt
    system_prompt = rubric.get_prompt()
    user_prompt = f"Please score this question:\n\n{prompt}"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    # Get the response from the model
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2  # Lower temperature for more consistent scoring
        )
        
        feedback = response.choices[0].message.content
        
        # Parse the scores from the feedback
        scores = {}
        for criterion in rubric.get_criteria_names():
            # Look for patterns like "Relevance: 4/5" or "Relevance - 4"
            import re
            pattern = rf"{criterion.capitalize()}:?\s*(\d+)(?:/\d+)?"
            match = re.search(pattern, feedback, re.IGNORECASE)
            
            if match:
                scores[criterion] = int(match.group(1))
            else:
                # If not found, default to middle score
                scores[criterion] = 3
                
        return {
            "scores": scores,
            "feedback": feedback,
            "question_id": question.get("id", "unknown"),
            "question_text": question["question"]
        }
        
    except Exception as e:
        print(f"Error scoring question: {e}")
        # Return default scores
        return {
            "scores": {criterion: 3 for criterion in rubric.get_criteria_names()},
            "feedback": f"Error: {str(e)}",
            "question_id": question.get("id", "unknown"),
            "question_text": question["question"]
        }


def score_dataset(client: OpenAI, dataset_path: str, output_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Score all questions in a dataset
    
    Args:
        client: OpenAI client
        dataset_path: Path to the dataset JSON file
        output_path: Path to save the scores (if None, will use dataset name + "_scores.json")

    Returns:
        List of score dictionaries
    """
    # Load the dataset
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    # Get the questions
    if "data" in dataset and "questions" in dataset["data"]:
        questions = dataset["data"]["questions"]
    elif "questions" in dataset:
        questions = dataset["questions"]
    else:
        raise ValueError("Could not find questions in the dataset")
    
    # Create the rubric
    rubric = ScoringRubric()
    
    # Score each question
    scores = []
    print(f"Scoring {len(questions)} questions...")
    for i, question in enumerate(tqdm(questions)):
        # Add an ID if not present
        if "id" not in question:
            question["id"] = f"q{i+1}"
            
        score = score_question(client, question, rubric)
        scores.append(score)
    
    # Save the scores if output_path is provided
    if output_path is None:
        # Generate output path based on dataset path
        base_name = os.path.splitext(dataset_path)[0]
        output_path = f"{base_name}_scores.json"
    
    # Add metadata
    result = {
        "metadata": {
            "dataset": dataset_path,
            "scoring_date": str(datetime.datetime.now()),
            "model": SCORER_MODEL,
            "num_questions": len(questions)
        },
        "scores": scores
    }
    
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Scores saved to {output_path}")
    return scores


def analyze_scores(scores: List[Dict[str, Any]], output_dir: Optional[str] = None):
    """
    Analyze and visualize the scores

    Args:
        scores: List of score dictionaries
        output_dir: Directory to save the visualizations
    """
    if not scores:
        print("No scores to analyze")
        return
    
    # Create output directory if it doesn't exist
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Extract scores into a DataFrame
    data = []
    for score_dict in scores:
        for criterion, score in score_dict["scores"].items():
            data.append({
                "question_id": score_dict["question_id"],
                "criterion": criterion,
                "score": score
            })
    
    df = pd.DataFrame(data)
    
    # Calculate summary statistics
    summary = df.groupby("criterion")["score"].agg(["mean", "std", "min", "max"]).reset_index()
    print("\nScore Summary:")
    print(summary)
    
    # Create visualizations
    
    # 1. Distribution of scores by criterion
    plt.figure(figsize=(12, 8))
    sns.boxplot(x="criterion", y="score", data=df)
    plt.title("Distribution of Scores by Criterion")
    plt.xlabel("Criterion")
    plt.ylabel("Score")
    if output_dir:
        plt.savefig(os.path.join(output_dir, "score_distribution.png"))
    
    # 2. Heatmap of average scores by question
    pivot_df = df.pivot_table(index="question_id", columns="criterion", values="score")
    plt.figure(figsize=(12, max(8, len(scores) * 0.4)))
    sns.heatmap(pivot_df, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Score'})
    plt.title("Scores by Question and Criterion")
    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, "score_heatmap.png"))
    
    # 3. Histogram of overall scores
    plt.figure(figsize=(10, 6))
    overall_scores = df.groupby("question_id")["score"].mean()
    sns.histplot(overall_scores, bins=10, kde=True)
    plt.title("Distribution of Average Question Scores")
    plt.xlabel("Average Score")
    plt.ylabel("Count")
    if output_dir:
        plt.savefig(os.path.join(output_dir, "overall_score_distribution.png"))
    
    # Show plots if not saving
    if not output_dir:
        plt.show()
    
    # Return the summary for further analysis
    return summary


def main(dataset_path: str, output_dir: Optional[str] = None, analyze: bool = True):
    """
    Main function to score a dataset and analyze the results
    
    Args:
        dataset_path: Path to the dataset JSON file
        output_dir: Directory to save the scores and visualizations
        analyze: Whether to analyze and visualize the scores
    """
    # Initialize OpenAI client
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    # Create output directory if it doesn't exist
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate output paths
    if output_dir:
        base_name = os.path.splitext(os.path.basename(dataset_path))[0]
        scores_path = os.path.join(output_dir, f"{base_name}_scores.json")
    else:
        base_name = os.path.splitext(dataset_path)[0]
        scores_path = f"{base_name}_scores.json"
    
    # Score the dataset
    scores = score_dataset(client, dataset_path, scores_path)
    
    # Analyze the scores if requested
    if analyze:
        analyze_scores(scores, output_dir)
    
    return scores


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Score and analyze evaluation questions")
    parser.add_argument("dataset", help="Path to the dataset JSON file")
    parser.add_argument("--output-dir", help="Directory to save the scores and visualizations")
    parser.add_argument("--no-analyze", action="store_true", help="Skip analysis and visualization")
    parser.add_argument("--api-key", help="OpenAI API key (if not set in environment)")
    
    args = parser.parse_args()
    
    # Set API key if provided
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key
    
    # Check if API key is available
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OpenAI API key not found. Please set the OPENAI_API_KEY environment variable or use --api-key.")
        exit(1)
    
    main(args.dataset, args.output_dir, not args.no_analyze)

