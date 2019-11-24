import networks
import models
import tr_helpers

animal_ai = {
    'NETWORK' : models.LSTMModelA2C(networks.animal_a2c_network_lstm6),
    'ENV_NAME' : 'AnimalAI',
}