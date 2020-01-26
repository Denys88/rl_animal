# Animal AI 1st Place Solution Presented on NIPS 2019 (my nickname is Trrrrr)
## Challenge website and results :http://animalaiolympics.com/results.html
## My NIPS 2019 Presentation [Animal AI Presentation.pdf](/Animal%20AI%20Presentation.pdf)

I used this exact code, and only made slight changes before sharing.
I added linux enviroment binaries by default. I tested everything on Linux only.
You might have some problems to run training on other platforms.

## Steps to Run My Code:
1) pip install -r [requirements.txt](/requirements.txt)
2) Run [download_networks.py](/download_networks.py) to download my networks. 
3) There are [hyperparams.py](/hyperparams.py) with all parameters which I used during training process. Please change BASE_DIR = '/home/trrrrr/Documents/github/ml/rl_animal' in hyperparams.py to your folder before run anything.


We were allowed to submit two networks and currently I don't know which one has won.

You can find different game configurations in [game_configurations.py](../blob/master/games_configurations.py)
You can find networks in [networks.py](/networks.py)
You can use [Player.ipynb](/Player.ipynb) to run my agents
To run my validation please use [Validation.ipynb](/Validation.ipynb)
To run train please use [test_a2c.ipynb](/test_a2c.ipynb)

There is chance that you will not be able to run my code with current parameters. Please try to reduce batch size and number of enviroments in this case.





