cd ..
export PYTHONPATH="$PYTHONPATH:$PWD" 
cd runnables 

export TF_CPP_MIN_LOG_LEVEL=2 # to ignore warnings about CPU/GPU

python leave_one_out.py

