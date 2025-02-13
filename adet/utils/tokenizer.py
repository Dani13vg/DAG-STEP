from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers
import os
import re

# Function to decode the tokens
def decode_tokens(tokenizer, token_ids):
    tokens = tokenizer.decode(token_ids)
    tokens = tokens.replace(" ##", "")  # This step is to handle any remaining artifacts
    return tokens

def preprocess_content(content):
    # Filter content to include only ASCII characters
    filtered_content = []
    for line in content:
        # Remove non-ASCII characters
        string = " ".join(line)
        filtered_line = re.sub(r'[^\x00-\x7F]+', '', string)
        filtered_content.append(filtered_line)
    return filtered_content

def train_tokenizer(content, save_path="tokenizer.json", vocab_size=224):
    # Check if tokenizer file exists

    if os.path.exists(save_path):
        tokenizer = Tokenizer.from_file(save_path)
        print("Tokenizer loaded from saved file.")
    else:
        # Preprocess content to limit it to ASCII or desired characters
        content = preprocess_content(content)
        
        # Initialize and train the tokenizer
        tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()  # Split by whitespace
        special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
        trainer = trainers.WordPieceTrainer(
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            min_frequency=5
        )
        tokenizer.train_from_iterator(content, trainer=trainer)

        # Save the trained tokenizer to the specified path
        tokenizer.save(save_path)
        print("Tokenizer trained and saved.")

    return tokenizer

if __name__ == "__main__":
    import sys
    import pdb
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from Dataloaders.faster_img_regex_dataloader import load_data

    # Dataset size
    data_size = '1.6M_L10'
    regex_string_path = f'Datasets/{data_size}_strings_regex_generated.json'

    # Load the data and train the tokenizer
    vocab, data = load_data(regex_string_path)
    #tokenizer_path = f'Tokenizers/{data_size}_tokenizer.json'
    tokenizer = train_tokenizer(vocab)

    # Print tokens for ASCII characters
    for i in range(128):
        print(f'{i}:', tokenizer.encode(chr(i)).ids, chr(i))

    # Print tokens for some special characters
    print(tokenizer.encode(r"\d").ids, r"\d")
    print(tokenizer.encode(r"\w").ids, r"\w")
    print(tokenizer.encode(r"[A-Z]+").ids, r"[A-Z]+")

    # Print decoded random token ids
    print(tokenizer.decode([69, 486, 491]))

    pdb.set_trace()