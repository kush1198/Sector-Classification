To use approach 1 or 2 run:

python run.py --dataset DATASET --glove_path GLOVE_PATH --word_dir WORD_DIR --featurization_type FEATURIZATION_TYPE --run_implementation RUN_IMPLEMENTATION

Here,

DATASET --> Path to dataset (pandas dataframe in .pkl format) with headers as "Requirements", "Requirements_clean" (requirements after basic cleaning) and "labels" (classes: Health, Safety, Entertainment, Energy and Other)

GLOVE_PATH --> Path to pre-trained glove embeddings

WORD_DIR --> Path to directory containing all special hand-crafted features, "\speciallists" in this case

FEATURIZATION_TYPE --> Type of featurization (neumeric) 0-->Average Glove, 1-->TF-IDF Glove, 2-->USE

RUN_IMPLEMENTATION --> run implementation number? (1/2)


To use approach 3 run (In Google colab to train on GPU):

https://colab.research.google.com/drive/1COk5MdKlhq4qi8HV4lIdNhwLyrwQeJF8?usp=sharing

