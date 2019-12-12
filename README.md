# Animal AI 1st Place Solution Presented on NIPS 2019 (my nickname is Trrrrr: http://animalaiolympics.com/results.html)
## Challenge website: http://animalaiolympics.com/
I used exact this code and only made slight changes before sharing.
I've added linux enviroment binaries by default. I tested everything on Linux only.
You might have some problems to run train on other platforms.

## Steps to Run My Code:
1) pip install -r requirements.txt
2) Run download_networks.py to download my networks. 
30) Please change BASE_DIR = '/home/trrrrr/Documents/github/ml/rl_animal' in hyperparams.py to your folder before run anything.
We were able to submit two networks and currently I don't know which one has won.
You can find different game configuration in game_configurations.py
There are  with all parameters which I used during training process.
You can find networks in networks.py.

### You can use Player.ipynb to run my agents
### To run my validation please use Validation.ipynb
### to run train please use test_a2c.ipynb

There is chance that you will not be able to run my code with current parameters. Please try to reduce batch size and number of enviroments in this case.





