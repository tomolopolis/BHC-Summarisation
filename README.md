
Source code repo for the analysis and experiments found in the paper: [Paper title]()

# Dataset Prep
- `./dataset_prep` : Scripts and notebooks for all M3 Dataset prep. Running MedCAT the pre-trained NEr+L model and aligning to raw input text.

# Extractive Model (Top-K) sentence Ranking
- `./extractive_approach` : The extractive approach, model training, run scripts etc.

# Abstractive Model Fine-Tuning
Our compute for CogStack data requires pre-built containers. I've included the required Dockerfile in the repo. to rebuild them use:

<pre>$ docker build -f Dockerfile.builder -t tsearle/summ_exp-base:latest . </pre>

This builds the base image, then use:

<pre>$ docker build . -t tsearle/summ_exp:latest</pre>

Once finished to run the container on all available GPU compute:

<pre>$ bash run_container.sh</pre>

If you just want to test the container on a CPU machine (i.e. a laptop) use:

<pre>$ bash run_container_cpu.sh</pre>

These run scripts mount two dirs, `./mimic_summ_data` and `./cg_summ_data/`. 

# Abstractive Model Fine-Tuning with Guidance Signal

You can use the pre-buil container again here. Open the `guidance_experiment_cfg/<mim3 or cg>/<.json>` file, and edit the `ds_path`
to point to your huggingface
<pre></pre>


# Acknowledgements
Huggingface is building some awesome libraries that enable this research.
- [Transformers](https://huggingface.co/docs/transformers/index)[Datasets](): 
- [DataSets](https://huggingface.co/docs/datasets/index)
- [Tokenizer](https://huggingface.co/docs/tokenizers/index)

Also PyTorch[] and Nvidia's CUDA obv. :-) 

MedCAT: docs, downloads and more on our clinical NER+L framework [here](https://github.com/CogStack/MedCAT). 

# Citation


