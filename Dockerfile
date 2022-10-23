FROM tsearle/summ_exp-base:latest

RUN python download_hf_assets.py "seq2seq" "t5-small"

RUN python download_hf_assets.py "seq2seq" "t5-base"

RUN python download_hf_assets.py "seq2seq" "facebook/bart-base"

RUN python download_hf_assets.py "seq2seq" "Kevincp560/distilbart-cnn-6-6-finetuned-pubmed"

RUN python download_hf_assets.py "seq2seq" "sshleifer/distilbart-xsum-9-6"

RUN python download_hf_assets.py "encoderDecoder" "emilyalsentzer/Bio_ClinicalBERT"

RUN python download_hf_assets.py "encoderDecoder" "emilyalsentzer/Bio_Discharge_Summary_BERT"

RUN python download_hf_assets.py "metric" "rouge"