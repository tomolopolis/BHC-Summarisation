# Dataset Cleaning and Prep Notebooks

- `Dataset_Prep.ipynb` - prepares the training data for both the extractive and abstractive training
- `Clean_Dataset.ipynb` - cleans and extracts relevant source notes, and discharge summary BHCs. Concatenates source notes in chronological order.
- `run_medcdat.py` - a script to run MedCAT over all notes output from `Clean_Dataset.ipynb`.