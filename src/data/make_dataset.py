# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from datasets import load_dataset
import pandas as pd
import transformers
from random import shuffle


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')


def assistant_fine_tuning_dataset():
    ds = load_dataset('birgermoell/open_assistant_dataset')
    df = ds['train'].to_pandas()
    return df


def chat_templatize_assistant_dataset(df: pd.DataFrame, tokenizer: transformers.PreTrainedTokenizer):
    # Define cleaning and preprocessing functions.
    def text_to_dialogue(text):
        return [sentence.replace('User:', '').replace('Chip:', '').strip() for sentence in text.split('Assistant:')]

    def dialogue_to_chat(dialogue):
        out = [{'role': 'system', 'content': 'You are a friendly chatbot assistant.'}]
        for idx, message in enumerate(dialogue):
            role = 'user' if idx % 2 == 0 else 'assistant'
            out.append({'role': role, 'content': message})
        return out

    def chat_to_input(chat):
        return tokenizer.apply_chat_template(chat, tokenize=False)

    def process_example(example):
        out = text_to_dialogue(example)  # Clean up data from redundant words
        out = dialogue_to_chat(out)  # Convert to ChatML format
        out = chat_to_input(out)  # Add a task description & Apply base model chat template
        return out

    data = list(map(process_example, df['text']))
    shuffle(data)

    # Tokenize data.
    tokenized_data = list(map(tokenizer, data))
    input_lengths = [len(x['input_ids']) for x in tokenized_data]
    split_idx = int(.99 * len(data))
    train_data, val_data = tokenized_data[:split_idx], tokenized_data[split_idx:]
    return train_data, val_data, input_lengths


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
