# Asbtractive Summaarisation Exp Config Dir
Various experiment configuration run files.

To run this on your own data you will need to point these experiment config .json files to your preprocessed datasets.
i.e. in experiment_cfg/mim3/bart.json, "preprocessed_dataset_path": "<your own dataset>"

NB: Datasets are pre-processed and saved to disk via Huggingface.datasets.save_to_disk, loading is hardcoded in the script via load_from_disk.

This dir results are the outputs of running:
<pre>
$ python run_summarization-bhc-dataset-pre-processed.py experiment_cfg/<cg OR mim3>/<model run .json>
</pre>  


