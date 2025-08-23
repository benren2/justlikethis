from huggingface_hub import hf_hub_download
import fasttext
import json
from transformers import AutoTokenizer, GPT2Tokenizer
from tqdm import tqdm

gpt2tokenizer = GPT2Tokenizer.from_pretrained("gpt2", use_fast=True)

def detokenize(token_ids: list[int], tokenizer: AutoTokenizer) -> str:
    """Detokenizes a list of token IDs into a cleaned string."""
    return tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()

def fasttext_predictive_strength(dataset: list[list[int]], topk: float):
    """Filters a dataset based on FastText model predictions."""
    print("Loading model...")
    try:
        model_filename = "PreSelect-classifier.bin"
        model_path = hf_hub_download("hkust-nlp/preselect-fasttext-classifier", model_filename)
        model = fasttext.load_model(model_path)
        print(f"Loaded FastText model from {model_path}")
    except Exception as e:
        print(f"Error loading FastText model: {e}")
        return []

    predictive_strength_results = []
    
    print("Calculating predictive strength...")
    for idx, doc_tokens in enumerate(tqdm(dataset, desc="Calculating Predictive Strength")):
        if not doc_tokens:
            predictive_strength_results.append({
                'original_tokens': doc_tokens,
                'predicted_labels': [],
                'predicted_probs': [],
                '_sort_key_value': 0.0
            })
            continue

        text = detokenize(doc_tokens, gpt2tokenizer)
        
        if not text.strip():
            predictive_strength_results.append({
                'original_tokens': doc_tokens,
                'predicted_labels': [],
                'predicted_probs': [],
                '_sort_key_value': 0.0
            })
            continue

        text = text.replace('\n', ' ')
        labels, probs = model.predict(text)
        
        label_prob_map = {l.replace('__label__', ''): p for l, p in zip(labels, probs)}
        prob_1 = label_prob_map.get('1', 0.0)

        predictive_strength_results.append({
            'original_tokens': doc_tokens,
            'predicted_labels': [l.replace('__label__', '') for l in labels],
            'predicted_probs': probs.tolist(),
            '_sort_key_value': prob_1
        })

    predictive_strength_results.sort(key=lambda item: item['_sort_key_value'], reverse=True)

    total_documents_to_keep = int(len(dataset) * min(max(topk, 0.0), 1.0))
    
    return [
        {
            'predicted_labels': item['predicted_labels'],
            'predicted_probs': item['predicted_probs'],
            'original_tokens': item['original_tokens']
        }
        for item in predictive_strength_results[:total_documents_to_keep]
    ]

def main():
    input_filename = "input.json"
    output_filename = "filtered.json"
    topk_fraction = 0.01

    print(f"Step 1: Loading dataset from `{input_filename}`...")
    with open(input_filename, "r", encoding="utf-8") as f:
        data = json.load(f)  # entire file is one array
        dataset = [list(obj["tokens"]) for obj in data]

    print(f"Loaded {len(dataset)} documents.")

    print("Step 2: Running FastText Predictive Strength filtering...")
    filtered_data_with_details = fasttext_predictive_strength(dataset, topk=topk_fraction)

    print(f"\nFiltering completed: Reduced from {len(dataset)} to {len(filtered_data_with_details)} documents.")

    print(f"\nStep 3: Saving filtered output to `{output_filename}`...")
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(filtered_data_with_details, f, ensure_ascii=False, indent=2)

    print(f"Done! Filtered data saved to `{output_filename}`.")
    print("\n--- FastText Filtering Process Finished ---")

if __name__ == "__main__":
    main()
