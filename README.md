# Multimodal Fallacy Classification in Political Debates

This repository contains the code and resources for the project "Multimodal Argument Mining: A Case Study in Political Debates," focusing on the exploration of audio's role in classifying argumentative fallacies in political debates.

## Dataset: MM-USED-fallacies

The repository includes the code required to generate the MM-USED-fallacies dataset. This dataset is created by leveraging resources from the [MM-USED](https://github.com/federicoruggeri/multimodal-am/tree/main/multimodal-dataset) dataset and the [USED-fallacy](https://github.com/pierpaologoffredo/IJCAI2022) dataset. By incorporating multimodal techniques, we aim to enrich the fallacy analysis beyond traditional text-based information, specifically by incorporating audio data from political debates.

## Repository Structure

The main directory includes the following subdirectories:

- `local_database/`: This directory contains the original datasets USED-fallacy and MM-DatasetFallacies. It includes three versions of the MM-DatasetFallacies dataset: `full`, `no_duplicates`, and `partial-used` (for debugging purposes).
- `resources/`: This directory contains additional resources for the experiments.
  - `clips_generation/`: This subdirectory contains the files necessary to generate audio clips corresponding to dialogues, snippets, and components.
  - `download/`: This subdirectory contains the files necessary to download the recordings used in the experiments.
- `results/`: This directory stores the results of the experiments.
- `runnables/`: This directory houses the scripts necessary to run the experiments.
- `utils/`: This directory contains utility files for the experiments.

Please refer to the specific directories for further details on their contents and usage.

## Usage

1. Dataset Generation:
   - To generate the MM-USED-fallacy dataset starting from MM-USED and USED-fallacy, follow the following steps. The code and resources necessary for the dataset generation are included. Please, note that the file containing all the annotations for the MM-USED-fallacy dataset is already included in the `local_database/MM-DatasetFallacies/full` directory under the name `dataset.csv`.
     - Download the recordings of the debates using the information provided in the `experiments/resources/download/download_links.csv` file. The recordings should saved under `resources/debates_audio_recordings`. Please, use whatever tool you prefer to obtain the recordings.
     - Run the `run_clips_generation.sh` script in the `runnables` directory create the audio clips corresponding to the dialogues, snippets, and components. The script will generate the MM-USED-fallacy dataset clips and store them under a new folder `local_database/MM-DatasetFallacies/audio_clips`.
     
2. Experiments:
   - To run the experiments in the paper and perform leave-one-debate-out cross-validation:
     - Open the script `runnables/leave_one_out.py` and set:
       - **_text_model_**: can have values _bert_, _roberta_, _sbert_
       - **_audio model_**: can have values _wav2vec_, _clap_
       - **_config_**: can have values _text_only_, _audio_only_, _text_audio_
     - Run the `run_leave_one_out.sh` script in the `runnables` directory. The script will run the experiments for each debate and store the results in the `results` directory.

Please refer to the specific directories for further details on their contents and usage.

## License

This project is licensed under the [CC.BY License](LICENSE). 


## Citing the work 
If using this dataset, please cite the following publication:

>   Eleonora Mancini, Federico Ruggeri, and Paolo Torroni. 2024. Multimodal Fallacy Classification in Political Debates. In Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 2: Short Papers), pages 170–178, St. Julian’s, Malta. Association for Computational Linguistics.

```bibtex
@inproceedings{mancini-etal-2024-multimodal,
    title = "Multimodal Fallacy Classification in Political Debates",
    author = "Mancini, Eleonora  and
      Ruggeri, Federico  and
      Torroni, Paolo",
    editor = "Graham, Yvette  and
      Purver, Matthew",
    booktitle = "Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 2: Short Papers)",
    month = mar,
    year = "2024",
    address = "St. Julian{'}s, Malta",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.eacl-short.16",
    pages = "170--178",
}

```