import sys

from datasets import load_metric
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel, EncoderDecoderConfig, \
    EncoderDecoderModel


def main():
    model_type = sys.argv[1]
    model_name = sys.argv[2]

    hf_cache_dir = './hf_cache_dir/'
    use_fast_tokenizer = False

    if model_type == 'seq2seq':
        print(f'Downloading Seq2Seq HF model:{model_name}')
        config = AutoConfig.from_pretrained(model_name, cache_dir=hf_cache_dir)
        AutoTokenizer.from_pretrained(model_name, cache_dir=hf_cache_dir, use_fast=use_fast_tokenizer)
        AutoModelForSeq2SeqLM.from_pretrained(model_name, config=config, cache_dir=hf_cache_dir)

    if model_type == 'encoderDecoder':
        # https://huggingface.co/transformers/model_doc/encoderdecoder.html
        print(f'Downloading EncoderDecoder HF Model:{model_name}')
        AutoConfig.from_pretrained(model_name, cache_dir=hf_cache_dir)
        AutoTokenizer.from_pretrained(model_name, cache_dir=hf_cache_dir)
        EncoderDecoderModel.from_encoder_decoder_pretrained(model_name, model_name)

    if model_type == 'metric':
        print(f'Downloading HF metrics:{model_name}')
        load_metric(model_name)


if __name__ == '__main__':
    main()
