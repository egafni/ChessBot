python -m pip install  torch==1.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html

see notebooks/dev.ipynb

Some todos:

* make a simple flask app to play (just a simple input form for the move and then retain the sequence of moves into as a GET url
* try different tokenizations (like just tokenizing the state of the board)
* try reward-to-go conditioning from decision transformer
* try a hugging face pretrained transformer, or a pytorch/pytorch lightning implementation (currently using Karpathy's mingpt)
* download/pre-process more games for a larger training dataset (currently have 250k games)
* write some infra to have it play a chess engine and get an ELO rating
