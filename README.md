
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
Huggingface, Meta and Nviai libraries enable this research:
- [Transformers](https://huggingface.co/docs/transformers/index)
- [DataSets](https://huggingface.co/docs/datasets/index)
- [Tokenizers](https://huggingface.co/docs/tokenizers/index)
- [PyTorch](https://pytorch.org/)
- [Cuda](https://developer.nvidia.com/cuda-toolkit)

MedCAT: docs, downloads and more on our clinical NER+L framework [here](https://github.com/CogStack/MedCAT). 

Get in touch with the CogStack team here: contact@cogstack.org

# Citation
@ARTICLE{Searle2022-bg,
  title         = "Summarisation of Electronic Health Records with Clinical
                   Concept Guidance",
  author        = "Searle, Thomas and Ibrahim, Zina and Teo, James and Dobson,
                   Richard",
  month         =  nov,
  year          =  2022,
  archivePrefix = "arXiv",
  eprint        = "2211.07126",
  primaryClass  = "cs.CL",
  arxivid       = "2211.07126"
}




