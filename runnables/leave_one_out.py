from utils import data_loader, converter, model_implementation, model_utils, reproducibility, evaluation, routine
import os
import numpy as np
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")

# WARNING: comment the following line if you are not using CUDA
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import tensorflow as tf
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
# config = ConfigProto(device_count = {'GPU': 1})
# #config = ConfigProto()
# config.gpu_options.allow_growth = True
# sess = InteractiveSession(config=config)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.set_visible_devices(gpus[0], 'GPU')
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)


# Reproducibility settings
seed = reproducibility.set_reproducibility()

# Model and Training Parameters
text_model = 'bert'
audio_model = 'wav2vec'
if audio_model == 'wav2vec':
    sample_rate = 16000
    max_frame_len = 768 # to check
elif audio_model == 'clap': 
    sample_rate = 48000 # 16000 for wav2vec, 48000 for clap
    max_frame_len = 512 # to check 



config = 'text_only'
config_params = {'text_model': text_model,
                'audio_model': audio_model,
                'sample_rate': sample_rate,
                'is_text_model_trainable': False, 
                'epochs': 500, #500
                'batch_size': 8, #8 
                'callbacks': 'early_stopping',
                'use_class_weights': True,
                'seed': seed,
                'lr':5e-05,
                'max_frame_len': max_frame_len, #512
                'config': config} #5e-05

                

#project_dir = os.path.normpath(os.path.dirname(os.path.abspath(__file__)))
project_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
df = data_loader.load(project_dir)

# get unique dialogue IDs
dialogue_ids = df['Dialogue ID'].unique()

# Create Run Path
run_path = model_utils.create_run_path(project_dir, 'leave_one_out', config)


# leave one out cross validation
results = routine.leave_one_out(dialogue_ids, df, project_dir, run_path, config, config_params, text_model, audio_model, sample_rate)

# Save Cross Validation Results
# TODO: save results of crossval for each config mode (text, audio, text_audio)

evaluation.avg_results_cross_validation(results, run_path, validation_strategy= 'leave_one_out',  config = config, save_results=True)
model_utils.save_config(run_path, config_params)