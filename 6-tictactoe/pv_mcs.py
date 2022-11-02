from game5 import State
from dual_network import DN_INPUT_SHAPE
from math import sqrt
from tensorflow.keras.models import load_model
from pathlib import Path
import numpy as np


PV_EVALUATE_COUNT = 50


def predict(model, state):
    a,b,c = DN_INPUT_SHAPE
    x = np.array([state.pieces, state.enemy_pieces])
    x = x.reshape(c,a,b).transpose(1,2,0).reshape(1,a,b,c)

    y = model.predict(x,batch_size=1)
    policies = y[0][0][list(state.legal_actions())]
    policies /= sum(policies) if sum(policies) else 1

    value = y[1][0][0]
    return policies, value
