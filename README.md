# Multimodal Fallacy Classification in Political Debates

This repository contains the code and resources for the project "Multimodal Fallacy Classification in Argument Mining," focusing on the exploration of audio's role in classifying argumentative fallacies in political debates.

## Dataset: MM-USED-fallacies

The repository includes the code required to generate the MM-USED-fallacies dataset. This dataset is created by leveraging resources from the [MM-USED](https://github.com/federicoruggeri/multimodal-am/tree/main/multimodal-dataset) dataset and the [USED-fallacy](https://github.com/pierpaologoffredo/IJCAI2022) dataset. By incorporating multimodal techniques, we aim to enrich the fallacy analysis beyond traditional text-based information, specifically by incorporating audio data from political debates.

## Repository Structure

- `data-creation/`: This directory contains the necessary resources and data files for generating the MM-USED-fallacies dataset.
- `experiments/`: This directory contains the necessary resources and data files for running the experiments.

The `experiments/` directory further includes the following subdirectories:

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
   - To generate the MM-USED-fallacy dataset starting from MM-USED and USED-fallacy, follow the instructions provided in the `data-creation/` directory. The code and resources necessary for the dataset generation are included.
  
2. Experiments:
   - To download the audio recordings for the political debates and create the audio clips corresponding to the dialogues, snippets, and components, run the `run_clips_generation.sh` script in the `experiments/runnables` directory. The script will first download the audio recordings in the `local_database/MM-DatasetFallacies/audio_clips` directory and then create the audio clips in the `experiments/resources/debates_audio_recordings` directory.
   - To run the experiments in the paper and perform leave-one-debate-out cross-validation:
     - Open the script `experiments/runnables/leave_one_out.py` and set:
       - **_text_model_**: can have values _bert_, _roberta_, _sbert_
       - **_audio model_**: can have values _wav2vec_, _clap_
       - **_config_**: can have values _text_only_, _audio_only_, _text_audio_
     - Run the `run_leave_one_out.sh` script in the `experiments/runnables` directory. The script will run the experiments for each debate and store the results in the `experiments/results` directory.

Please refer to the specific directories for further details on their contents and usage.

## License

This project is licensed under the [MIT License](LICENSE). Feel free to use the code and
