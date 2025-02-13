from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import json
import re
import time
from groq import Groq
from adet.utils.queries import indices_to_text

models = [
'gemma2-9b-it',
'llama-3.3-70b-versatile',
'llama-3.1-8b-instant',
'llama3-70b-8192',
'llama3-8b-8192',
'mixtral-8x7b-32768',
'gemma2-9b-it',
'llama-3.3-70b-versatile',
'llama-3.1-8b-instant',
'llama3-70b-8192',
'llama3-8b-8192',
'mixtral-8x7b-32768',
]


def generate_regexes(sentence, model, tokenizer, device, use_groq=False, i=0):
    """Generate regexes for a given sentence using the chosen model."""
    valid = False
    if use_groq:
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        messages = [{"role": "user", "content": f"Generate three DIFFERENT regular expressions in different lines, without bullet points and without explaining the result, that matches the following sentence and nothing more: '{sentence.strip()}'"}]
        try:
            chat_completion = client.chat.completions.create(messages=messages, model=models[i // 990])
    
        except Exception as e:
            raise
        
        regexes = chat_completion.choices[0].message.content
        return regexes.strip().split('\n')
    
    while not valid:
        messages = [{"role": "user", "content": f"Generate three DIFFERENT regular expressions in different lines, without bullet points and without explaining the result, that matches the following sentence and nothing more: '{sentence.strip()}'"}]
        
        encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
        model_inputs = encodeds.to(device)
        generated_ids = model.generate(model_inputs, max_new_tokens=200, do_sample=True)
        decoded = tokenizer.batch_decode(generated_ids)
        regexes = decoded[0].split('[/INST]')[1]
        regex_list = format_regex(regexes, sentence)
        
        if len(regex_list) > 0:
            valid = True
        else:
            print(f"Invalid regex output: {regexes}\n")
    
    return regex_list

def remove_duplicate_regexes(regexes):
    """Remove duplicate regexes from the list."""
    return list(set(regexes))

def convert_sentence_to_regex(sentence):
    """Convert a sentence to a regex."""
    return re.escape(sentence)
    

def format_regex(regex, sentence):
    """Ensure regex formatting and return a list of valid regexes that match the sentence."""
    lines = regex.strip().split('\n')
    clean_lines = []
    invalid_regexes = 0
    
    for line in lines:
        line = line.strip().replace("`", "")
        if line:
            try:
                pattern = re.compile(line)
                if pattern.fullmatch(sentence):
                    clean_lines.append(line)
                else:
                    invalid_regexes += 1
            except re.error:
                invalid_regexes += 1
    
    return remove_duplicate_regexes(clean_lines[:3]), invalid_regexes

def process_json(json_path, output_path, model, tokenizer, device, use_groq=False):
    """Process the input JSON file, generate regexes, and save a new JSON file with regexes included."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    total_sentences = 0
    total_processed_sentences = 0
    sentences_with_3_regex = 0
    sentences_with_2_regex = 0
    sentences_with_1_regex = 0
    sentences_with_0_regex = 0
    total_invalid_regexes = 0

    initial_time = time.time()

    stats_output_path = os.path.splitext(output_path)[0] + "_stats.txt"
    
    for i, annotation in enumerate(data["annotations"]):

        sentence = indices_to_text(annotation["rec"])
        
        total_sentences += 1
        if len(re.findall(r'[^a-zA-Z]', sentence)) >= 2:
            total_processed_sentences += 1
            try:
                regexes = generate_regexes(sentence, model, tokenizer, device, use_groq, i=i)

            except Exception as e:
                print(f"Error occurred: {e}")
                output_path = os.path.splitext(output_path)[0] + f"_{i}.json"
                break

            regexes, invalid_regexes = format_regex('\n'.join(regexes), sentence)
            total_invalid_regexes += invalid_regexes
            
            if len(regexes) > 0:
                annotation["regex"] = regexes
            
            if len(regexes) == 3:
                sentences_with_3_regex += 1
            elif len(regexes) == 2:
                sentences_with_2_regex += 1
            elif len(regexes) == 1:
                sentences_with_1_regex += 1
            else:
                sentences_with_0_regex += 1
                annotation["regex"] = [convert_sentence_to_regex(sentence)]

        else:
            annotation["regex"] = [convert_sentence_to_regex(sentence)]

        if total_sentences % 100 == 0:
            print(f"Processed {total_sentences} sentences in {time.time() - initial_time:.2f} seconds.")
            print(f"Total sentences processed: {total_processed_sentences}")
            print(f"Sentences with 3 regexes: {sentences_with_3_regex}")
            print(f"Sentences with 2 regexes: {sentences_with_2_regex}")
            print(f"Sentences with 1 regex: {sentences_with_1_regex}")
            print(f"Sentences with 0 regex: {sentences_with_0_regex}")
            print(f"Total invalid regexes: {total_invalid_regexes}")
            print(f"Model: {models[i // 990]}")
            print(f'Current sentence: "{sentence}" and regexes: {regexes}\n')

            with open(output_path, 'w') as f:
                json.dump(data, f, indent=4)

            with open(stats_output_path, 'w') as stats_file:
                stats_file.write(f"Total number of sentences in the document: {total_sentences}\n")
                stats_file.write(f"Total sentences processed: {total_processed_sentences}\n")
                stats_file.write(f"Sentences with 3 regexes: {sentences_with_3_regex}\n")
                stats_file.write(f"Sentences with 2 regexes: {sentences_with_2_regex}\n")
                stats_file.write(f"Sentences with 1 regex: {sentences_with_1_regex}\n")
                stats_file.write(f"Sentences with 0 regex: {sentences_with_0_regex}\n")
                stats_file.write(f"Total invalid regexes: {total_invalid_regexes}\n")
                stats_file.write(f"Time taken: {time.time() - initial_time:.2f} seconds\n")

    print(f"Total number of sentences in the document: {total_sentences}")
    print(f"Total sentences processed: {total_processed_sentences}")
    print(f"Sentences with 3 regexes: {sentences_with_3_regex}")
    print(f"Sentences with 2 regexes: {sentences_with_2_regex}")
    print(f"Sentences with 1 regex: {sentences_with_1_regex}")
    print(f"Sentences with 0 regex: {sentences_with_0_regex}")
    print(f"Total invalid regexes: {total_invalid_regexes}")

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)
    
    with open(stats_output_path, 'w') as stats_file:
        stats_file.write(f"Total number of sentences in the document: {total_sentences}\n")
        stats_file.write(f"Total sentences processed: {total_sentences}\n")
        stats_file.write(f"Sentences with 3 regexes: {sentences_with_3_regex}\n")
        stats_file.write(f"Sentences with 2 regexes: {sentences_with_2_regex}\n")
        stats_file.write(f"Sentences with 1 regex: {sentences_with_1_regex}\n")
        stats_file.write(f"Sentences with 0 regex: {sentences_with_0_regex}\n")
        stats_file.write(f"Total invalid regexes: {total_invalid_regexes}\n")


# Function to read a JSON file and remove all the annotations that does not contain a regex
def filter_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    print("Filtering file...")
    initial_time = time.time()
    
    for i, annotation in enumerate(data["annotations"]):

        sentence = indices_to_text(annotation["rec"])
        
        if "regex" not in annotation:
            annotation["regex"] = [convert_sentence_to_regex(sentence)]
        
        elif len(annotation["regex"]) == 0:
            annotation["regex"] = [convert_sentence_to_regex(sentence)]

        if i % 10000 == 0:
            print(f"Processed {i} sentences in {time.time() - initial_time:.2f} seconds.")

    print(f"Total number of sentences in the document: {len(data['annotations'])}")

    output_path = os.path.splitext(file_path)[0] + "_filtered.json"
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        filter_file(sys.argv[1])

    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        assert torch.cuda.is_available()
        device = "cuda"
        
        use_groq = True # bool(int(os.environ.get("USE_GROQ", "0")))
        
        print(f"Using GROQ: {use_groq}")
        if not use_groq:
            model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", torch_dtype=torch.float16)
            model.to(device)
            tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
        else:
            model = None
            tokenizer = None
        
        input_json_path = "datasets/hiertext/train.jsonl"
        output_json_path = "datasets/hiertext/train_with_regex1.json"
        
        process_json(input_json_path, output_json_path, model, tokenizer, device, use_groq)

        print(f"Finished processing JSON file. Saved output to {output_json_path} with ")
