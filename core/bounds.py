import numpy as np
import torch

from core.kl_inv import klInvFunction
from models.majority_vote import MultipleMajorityVote

def mcallester_bound(n, model, risk, delta, coeff=1, order = None, div='KL', type=None, verbose=False, monitor=None):

    if type == 'des_rivasplata_dir':
        kl = model.KL_disintegrated()
    elif div == 'KL':
        kl = model.KL()
    elif div == 'Renyi':
        kl = model.Renyi(order)

    if isinstance(model, MultipleMajorityVote): # informed priors
        
        const = np.log(4 * (n**2 / 4)**0.5 / delta)
        kl *= 2

    else:
        const = np.log(2 * (n**0.5) / delta)

    bound = coeff * (risk + ((kl + const) / 2 / n)**0.5)

    if monitor:
        monitor.write(train={"KL": kl.item(), "risk": risk.item()})

    return bound 

def seeger_bound(n, model, risk, delta, coeff=1, order = None, div='KL', type=None, verbose=False, monitor=None):

    if type == 'des_rivasplata_dir':
        kl = model.KL_disintegrated()
    elif div == 'KL':
        kl = model.KL() 
    elif div == 'Renyi':
        kl = model.Renyi(order)
        

    if isinstance(model, MultipleMajorityVote): # informed priors
        kl *= 2
        if div == 'Renyi':
           
            const = 2*((2 * order - 1) / (order - 1) * np.log(4 / delta)) + np.log(2 * n)
        else:
            const = 2 * np.log((4 * (n/2)**0.5)/ delta) 
    else:
        if div == 'Renyi':
            const = (2 * order - 1) / (order - 1) * np.log(2 / delta) + np.log(2 * n ** 0.5)
        else:
            const = np.log(2 * (n ** 0.5) / delta)

    bound = coeff * klInvFunction.apply(risk, (torch.max(kl + const, torch.tensor(0))  / n))


    if monitor:
        monitor.write(train={"KL": kl.item(), "risk": risk.item()})

    return bound 


BOUNDS = {
    "mcallester": mcallester_bound,
    "seeger": seeger_bound,
}
