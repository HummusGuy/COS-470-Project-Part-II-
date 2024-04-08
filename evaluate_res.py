import pandas as pd
import nltk
import re
from nltk.translate.bleu_score import sentence_bleu
from sentence_transformers import SentenceTransformer, util

# For cosine sim
model = SentenceTransformer("all-mpnet-base-v2")


def import_tsv(file_path):
    return pd.read_csv(file_path, sep='\t')


# In results get each identified difficult term and clean them up
def extract_terms(output):
    terms = re.findall(r"(?<!\d)\b.*?(?=\n|$)", output)
    terms = [term.strip() for term in terms if term.strip() and not term.strip().isdigit()]

    terms = [re.sub(r'\d+\.\s*', '', term).strip() for term in terms]


    return terms


def calculate_cosine_similarity(generated_definition, ground_truth_definition):
    
    embeddings1 = model.encode(generated_definition, convert_to_tensor=True)
    embeddings2 = model.encode(ground_truth_definition, convert_to_tensor=True)

    cosine_scores = util.cos_sim(embeddings1, embeddings2)

    return cosine_scores[0][0]




# In results get each definition for the term
def parse_output(output):
    pattern = r'\d+\.\s+(.*?)(?=\d+\.\s+|\Z)'

    definitions = re.findall(pattern, output, re.DOTALL)
    
    return definitions


def main():
    results_data = import_tsv("initial_results.tsv")
    ground_truths_data = import_tsv("ground_truths.tsv")
    
    terms = ground_truths_data["term"].tolist()
    snt_ids = results_data["snt_id"].tolist()
    passages = results_data["Passage"].tolist()

    ground_truth_definitions = ground_truths_data["definition"].tolist()
    terms_to_ground_truth = {}
    
    snt_id_to_passage = {}
    for i, snt_id in enumerate(snt_ids):
        snt_id_to_passage[snt_id] = passages[i]

    ground_truth_paper_to_terms = {}
    for _, row in ground_truths_data.iterrows():
        snt_id = row['snt_id']
        term = row['term']
        if snt_id in ground_truth_paper_to_terms:
            ground_truth_paper_to_terms[snt_id].append(term)
        else:
            ground_truth_paper_to_terms[snt_id] = [term]
    
    
    for i, term in enumerate(terms):
        terms_to_ground_truth[term.lower()] = ground_truth_definitions[i].lower()

    
    # Get difficult term predictions and definitions from each paper
    paper_to_definitions = {}
    paper_to_terms = {}
    for index, row in results_data.iterrows():
        predicted_definitions = parse_output(row["Explanations"])
        predicted_terms = extract_terms(row["Identified Difficult Terms"])
        paper_to_definitions[row["snt_id"]] = predicted_definitions
        paper_to_terms[row["snt_id"]] = predicted_terms

    snt_id_list = []  
    passage_list = [] 
    predicted_terms_list = []  
    actual_terms_list = []   
    predicted_definitions_list = []   
    actual_definitions_list = []   
    cosine_similarity_list = [] 
    
    for paper, definitions in paper_to_definitions.items():
        # Load all data for a pd data frame for tsv
        ground_truth_definitions = [] # Gather all ground truths 4 batch of terms
        cos_sims = [] # To gather in batches of 5 for displaying
        terms = paper_to_terms[paper]
        predicted_terms_list.append(terms)
        actual_terms_list.append(ground_truth_paper_to_terms.get(paper, []))
        snt_id_list.append(paper)
        passage_list.append(snt_id_to_passage[paper])
        
        # Compare definition to ground truth
        for i, definition in enumerate(definitions):
            if i < 5: # If term correctly identified for a paper
                if terms[i].lower() in terms_to_ground_truth and terms[i].lower() in ground_truth_paper_to_terms.get(paper, []):
                    ground_truth_definitions.append(terms_to_ground_truth[terms[i].lower()])
                    similarity = calculate_cosine_similarity(definition, terms_to_ground_truth[terms[i].lower()])
                    cos_sims.append(similarity)
                else:
                    cos_sims.append("Not a properly extracted complex term")
        
        cosine_similarity_list.append(cos_sims)
        predicted_definitions_list.append(definitions)
        actual_definitions_list.append(ground_truth_definitions)
   
    # Panda data frame for evaluated results
    data = {
        'snt_id': snt_id_list,
        'passage': passage_list,
        'predicted_complex_terms': predicted_terms_list,
        'actual_complex_terms': actual_terms_list,
        'predicted_definitions': predicted_definitions_list,
        'actual_definitions': actual_definitions_list,
        'cosine_similarity': cosine_similarity_list
    }   
    df = pd.DataFrame(data)

    df.to_csv('evaluated_results.tsv', sep='\t', index=False)
    
    # Convert DataFrame to LaTeX table
    latex_table = df.to_latex(index=False, escape=False)

    with open('evaluated_results.tex', 'w') as f:
        f.write(latex_table)

main()
