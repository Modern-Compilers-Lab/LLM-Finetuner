import pandas as pd
from datasets import load_dataset,Dataset ,DatasetDict,load_from_disk
import pandas as pd
import re
from nltk.translate.bleu_score import corpus_bleu
from rouge_score import rouge_scorer
from transformers import BertTokenizer, BertForMaskedLM, BertModel
from bert_score import BERTScorer
from prometheus_eval import PrometheusEval
from prometheus_eval.prompts import ABSOLUTE_PROMPT, SCORE_RUBRIC_TEMPLATE
import yaml
from huggingface_hub import login
print("Imported libs \n")



# Define a function to load parameters from a YAML file
def load_config_from_yaml(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config



# Load parameters from YAML file
config = load_config_from_yaml("evaluate/evaluate_config.yaml")



# Access parameters
output_file_path = config["output_file_path"]
dataset_path = config["dataset_path"]

print("Loaded parameters \n")



#Code begins


dataset_p=load_from_disk(dataset_path)
# dataset_p = load_dataset(dataset_path)
print("Dataset loaded \n")


machine_results_Finetuned= list(dataset_p["finetuned_answers"])
reference_texts=list(dataset_p["output"])


print(f"An exampel of the model output (10th instruction): \n {machine_results_Finetuned[9]} \n")

def extract_text(results):
    pattern = re.compile(r'Response\s*:\s*(.*)')
    extracted_texts = []
    count = 0
    for text in results:
        match = re.search(pattern, text)
        if match:
            extracted_texts.append(match.group(1))
        else:
            extracted_texts.append("")
            count += 1
    print(f"Pattern not found for {count} results.")
    return extracted_texts



def calculate_bleu_score(machine_results, reference_texts, output_file):
    bleu_score = corpus_bleu([[ref.split()] for ref in reference_texts], [gen.split() for gen in machine_results])
    result = f'BLEU Score: {bleu_score}'
    print(result)
    with open(output_file, 'a') as file:
        file.write(result + '\n')

def calculate_rouge_scores(generated_answers, ground_truth, output_file):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    total_rouge1, total_rouge2, total_rougeL = 0, 0, 0
    for gen, ref in zip(generated_answers, ground_truth):
        scores = scorer.score(gen, ref)
        total_rouge1 += scores['rouge1'].fmeasure
        total_rouge2 += scores['rouge2'].fmeasure
        total_rougeL += scores['rougeL'].fmeasure
    average_rouge1 = total_rouge1 / len(generated_answers)
    average_rouge2 = total_rouge2 / len(generated_answers)
    average_rougeL = total_rougeL / len(generated_answers)
    result = (f'Average ROUGE-1: {average_rouge1}\n'
              f'Average ROUGE-2: {average_rouge2}\n'
              f'Average ROUGE-L: {average_rougeL}')
    print(result)
    with open(output_file, 'a') as file:
        file.write(result + '\n')

def calculate_bert_score(generated_answers, ground_truth, output_file):
    scorer = BERTScorer(model_type='bert-base-uncased')
    P, R, F1 = scorer.score(generated_answers, ground_truth)
    results = []
    for i, (p, r, f1) in enumerate(zip(P, R, F1)):
        result = (f"Pair {i + 1} - BERTScore Precision: {p.mean():.4f}, "
                  f"Recall: {r.mean():.4f}, F1: {f1.mean():.4f}")
        results.append(result)
        print(result)
    avg_precision = sum(p.mean() for p in P) / len(P)
    avg_recall = sum(r.mean() for r in R) / len(R)
    avg_f1 = sum(f1.mean() for f1 in F1) / len(F1)
    avg_result = (f"\nAverage BERTScore - Precision: {avg_precision:.4f}, "
                  f"Recall: {avg_recall:.4f}, F1: {avg_f1:.4f}")
    results.append(avg_result)
    print(avg_result)
    with open(output_file, 'a') as file:
        file.write('\n'.join(results) + '\n')





# Extract text from results (i.e we will extract everything after "Response :")
machine_results_Finetuned_Copy = extract_text(machine_results_Finetuned)

print("Responses extracted \n")

print(f"An exampel of extracted response (10th instruction): \n {machine_results_Finetuned_Copy[9]} \n")

print(f"An exampel of reference response (10th instruction): \n {machine_results_Finetuned_Copy[9]} \n")

# Calculate BLEU score
print("BLEU Score for Finetuned:")
calculate_bleu_score(machine_results_Finetuned_Copy, reference_texts,output_file_path)

# Calculate ROUGE scores
print("ROUGE Scores for Finetuned:")
calculate_rouge_scores(machine_results_Finetuned_Copy, reference_texts,output_file_path)


# Calculate BERTScore
print("BERTScores for Finetuned:")
calculate_bert_score(machine_results_Finetuned_Copy, reference_texts,output_file_path)






instructions_text=list(dataset_p["instruction"])








print("LLM evaluation begins \n")

judge = PrometheusEval(model_id="prometheus-eval/prometheus-7b-v2.0", absolute_grade_template=ABSOLUTE_PROMPT)



rubric={
  "criteria":"Is the answer both factually accurate, matching the reference answer, and presented in a clear and concise manner that's easy  to understand. ?",
  "score1_description":"If the generated answer is not relevant to the user query and reference answer.",
  "score2_description":"If the generated answer is correct according to reference answer but not relevant to user query.",
  "score3_description":"If the generated answer is relevant to the user query and correct according to reference answer but has some mistakes in facts.",
  "score4_description":"If the generated answer is relevant to the user query and has the exact same metrics and correct as the reference answer, but it is not as concise.",
  "score5_description":"If the generated answer is relevant to the user query and fully correct according to the reference answer."
}



feedbacks, scores = judge.absolute_grade(
    instructions=instructions_text,
    responses=machine_results_Finetuned_Copy,
    params={},
    rubric=rubric,
    reference_answers=reference_texts
)
print(f"An exampel of the score attributed to a responce (10th instruction):  {scores[9]} \n")
print(f"An exampel of the feedback attributed to a responce (10th instruction): \n {feedbacks[9]} \n")

# Calculate and print average score
average_score = sum(scores) / len(scores)
print(f"Average Score: {average_score:.4f} \n")

# Calculate and print percentage of scores that are 4 or 5
num_high_scores = sum(1 for score in scores if score == 4 or score == 5)
percentage_high_scores = (num_high_scores / len(scores))
print(f"Ratio of Scores 4 or 5: {percentage_high_scores:.4f} \n")

# Write results to the output file
with open(output_file_path, 'a') as file:
    file.write(f"Average Score: {average_score:.4f}\n")
    file.write(f"Ratio of Scores 4 or 5: {percentage_high_scores:.4f}\n")