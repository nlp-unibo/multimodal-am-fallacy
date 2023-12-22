cd ..
export PYTHONPATH="$PYTHONPATH:$PWD" 
cd runnables 

export TF_CPP_MIN_LOG_LEVEL=2 # to ignore warnings about CPU/GPU
#python leave_one_out.py


# Generate clips if local_database/MM-DatasetFallacies/audio_clips is empty
python clips_generation.py

