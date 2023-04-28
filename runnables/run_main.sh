cd ..
cd utils
export PYTHONPATH="$PYTHONPATH:$PWD" 
cd ..
cd runnables 

export TF_CPP_MIN_LOG_LEVEL=2 # to ignore warnings about CPU/GPU
python main.py