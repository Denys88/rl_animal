import networks
import models
import tr_helpers


animal_ai = {
    'NETWORK' : models.LSTMModelA2C(networks.animal_a2c_network_lstm2),
    'ENV_NAME' : 'AnimalAI',
}