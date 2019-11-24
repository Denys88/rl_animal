import players
import env_configurations

validation_list = [
    '/home/trrrrr/Documents/github/ml/rl_animal/validation/1-Food.yaml',
    '/home/trrrrr/Documents/github/ml/rl_animal/validation/2-Preferences.yaml',
    '/home/trrrrr/Documents/github/ml/rl_animal/validation/3-Obstacles.yaml',
    '/home/trrrrr/Documents/github/ml/rl_animal/validation/4-Avoidance.yaml',
    '/home/trrrrr/Documents/github/ml/rl_animal/validation/5-SpatialReasoning.yaml',
    '/home/trrrrr/Documents/github/ml/rl_animal/validation/6-Generalization.yaml',
    '/home/trrrrr/Documents/github/ml/rl_animal/validation/1-Food500.yaml',
    '/home/trrrrr/Documents/github/ml/rl_animal/validation/2-Preferences500.yaml',
    '/home/trrrrr/Documents/github/ml/rl_animal/validation/3-Obstacles500.yaml',
    '/home/trrrrr/Documents/github/ml/rl_animal/validation/4-Avoidance500.yaml',
    '/home/trrrrr/Documents/github/ml/rl_animal/validation/5-SpatialReasoning500.yaml',
    '/home/trrrrr/Documents/github/ml/rl_animal/validation/6-Generalization500.yaml',
]

validation_list500 = [

]

class Validator:
    def __init__(self, sess, config, path):
        self.run_count = 100
        self.seed = 32
        #self.seed = 0
        self.config = config
        self.player = players.PpoPlayerDiscrete(sess, config)
        self.player.restore(path)
        self.env = env_configurations.create_animal(1, inference = False, config = validation_list[0], seed = self.seed)
        

    def run(self):
        from animalai.envs.arena_config import ArenaConfig
        reward = 0
        total_loose = 0
        for val_config in validation_list:
            print('starting: ', val_config)
            s =  self.env.reset(ArenaConfig(val_config))
            cr = 0
            c_looses = 0
            for _ in range(self.run_count):
                self.player.reset()
                ep_r = 0
                for it in range(5000):
                    action = self.player.get_action(s, False)
                    s, r, done, _ =  self.env.step(action)
                    cr += r
                    ep_r += r
                    if done:
                        s = self.env.reset()
                        break

                if ep_r < -0.95:
                    c_looses += 1
                    
            print(val_config, "reward: ", cr)  
            print("c_looses: ", c_looses)  
            reward = reward + cr / float(self.run_count)     
            total_loose = total_loose + c_looses / float(self.run_count)     
        return reward, total_loose
