 cd /data/users/k1897038/mimic_summarisation/extractive_approach

##### S-BERT embeddings with M-3 ######
 python lstm_train.py -sl 1 -bs 200 -en sl1bs200 -e 10

 python lstm_train.py -sl 2 -bs 200 -en sl2bs200 -e 10

python lstm_train.py -sl 10 -bs 100 -en sl10bs100 -cp sl10bs100/checkpoints/chk--epoch_0-steps_200.pt >

 python lstm_train.py -sl 10 -bs 200 --checkpoint_steps 50 \
     -en sl10bs200-bidir-100hd --cuda_device 2 \
     -cp sl10bs200-bidir-100hd/checkpoints/chk-epoch_5-final \
     -e 10


 python lstm_train.py -sl 5 -bs 50 --checkpoint_steps 200 \
     -en sl5bs50 --cuda_device 1 \
     -e 10

 python lstm_train.py  -sl 2 -bs 200 --checkpoint_steps 50 \
     -en sl2bs200 --cuda_device 1 \
     -e 10
    
python lstm_train.py  -sl 5 -bs 200 --checkpoint_steps 50 \
    -en sl2bs200 --cuda_device 2 \
    -e 10

######## GloVe Embedding Runs ######

 python lstm_train.py -sl 1 -bs 200 -en sl1_spacy_bs200 -e 10 --use_sbert_embeddings 0

 python lstm_train.py -sl 2 -bs 200 -en sl2_spacy_bs200 -e 10 --use_sbert_embeddings 0 --cuda_device 2

 python lstm_train.py -sl 3 -bs 200 -en sl3_spacy_bs200 -e 10 --use_sbert_embeddings 0 --cuda_device 2

 python lstm_train.py -sl 5 -bs 200 -en sl5_spacy_bs200 -e 10 --use_sbert_embeddings 0 --cuda_device 2

 python lstm_train.py -sl 10 -bs 200 -en sl10_spacy_bs200 -e 10 --use_sbert_embeddings 0 --cuda_device 0

 python lstm_train.py -sl 15 -bs 200 -en sl15_spacy_bs200 -e 10 --use_sbert_embeddings 0 --cuda_device 1


#### S-BERT embeddings for CogStack data #####
 python lstm_train.py -sl 1 -bs 50 -en cg-sl1bs50 -e 10 --is_cg 1

 python lstm_train.py -sl 2 -bs 50 -en cg-sl2bs50 -e 10 --is_cg 1

 python lstm_train.py -sl 3 -bs 50 -en cg-sl3bs50 -e 10 --is_cg 1

 python lstm_train.py -sl 5 -bs 50 -en cg-sl5bs50 -e 10 --is_cg 1

 python lstm_train.py -sl 10 -bs 50 -en cg-sl10bs50 -e 10 --is_cg 1

 python lstm_train.py -sl 15 -bs 50 -en cg-sl15bs50 -e 10 --is_cg 1

### wv embeddings for CogStack data #####

 python lstm_train.py -sl 1 -bs 50 -en cg-sl1bs50-wv -e 10 --is_cg 1 --use_sbert_embeddings 0

 python lstm_train.py -sl 2 -bs 50 -en cg-sl2bs50-wv -e 10 --is_cg 1 --use_sbert_embeddings 0

 python lstm_train.py -sl 3 -bs 50 -en cg-sl3bs50-wv -e 10 --is_cg 1 --use_sbert_embeddings 0

 python lstm_train.py -sl 5 -bs 50 -en cg-sl5bs50-wv -e 10 --is_cg 1 --use_sbert_embeddings 0

 python lstm_train.py -sl 10 -bs 50 -en cg-sl10bs50-wv -e 10 --is_cg 1 --use_sbert_embeddings 0

 python lstm_train.py -sl 15 -bs 50 -en cg-sl15bs50-wv -e 10 --is_cg 1 --use_sbert_embeddings 0

