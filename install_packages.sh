pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install pytorch_lightning==1.0.6
pip install GitPython
pip install rouge_score sacrebleu
pip install sentencepiece

#f'You have asked for AMP support {amp_type}, but there is no support on your side yet.'
# ModuleNotFoundError: You have asked for AMP support apex, but there is no support on your side yet. Consider installing torch >= 1.6 or NVIDIA Apex.