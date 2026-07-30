import torch
import numpy as np

from torch.distributions.dirichlet import Dirichlet as Dir
from torch.distributions.multivariate_normal import MultivariateNormal as Gaus
from torch.distributions.log_normal import LogNormal as LogGaus
from torch import lgamma, digamma

from core.utils import BetaInc


import torch

def log_Beta(vec):
    """
    Logarithm of the beta function.
    """
    return lgamma(vec).sum() - lgamma(vec.sum())

class Dirichlet():

    def __init__(self, alpha, mc_draws=1):

        self.alpha = alpha 
        self.mc_draws = mc_draws
        self.thetas =None 

    # Kullback-Leibler divergence between two Dirichlets
    def KL(self, beta):

        exp_alpha = torch.exp(self.alpha)  
         

        res = lgamma(exp_alpha.sum()) - lgamma(exp_alpha).sum()
        res -= lgamma(beta.sum()) - lgamma(beta).sum()
        res += torch.sum((exp_alpha - beta) * (digamma(exp_alpha) - digamma(exp_alpha.sum())))


        return res
        
    def KL_disintegrated(self, beta):
        # KL divergence between two dirichlet distributions, with all the probability mass on a single point
        alphas = torch.exp(self.alpha)

        
        res = Dir(alphas).log_prob(self.thetas)
        res -= Dir(beta).log_prob(self.thetas)
        res = torch.clamp(res.squeeze(0), 0)
        return res

    
    def Renyi(self, beta, order):
        if order <= 1 or torch.isclose(torch.tensor(order), torch.tensor(1.0)):
            raise ValueError("order doit être > 1 et != 1")

        self.order = order
        self.beta = beta 


        
        
        alphas = torch.exp(self.alpha)      
        
        if (alphas <= 0).any() or (beta <= 0).any():
            raise ValueError("Every alphas and betas must be > 0")
        
        

        logB_alpha = log_Beta(alphas)
        logB_beta  = log_Beta(beta)
        logB_mix   = log_Beta(order * alphas + (1 - order) * beta)

        res = logB_beta - logB_alpha + (logB_mix - logB_alpha)/(order-1)


        return res

    
    
    def risk(self, batch, risk_type="stochastic", mean=True, post=None):

        y_target, y_pred = batch


        

        exp_alpha = torch.exp(self.alpha)




        if risk_type == "stochastic":
            exp_alpha = torch.exp(self.alpha)
            
            correct = torch.where(y_target == y_pred, exp_alpha, torch.zeros(1)).sum(1)
            wrong = torch.where(y_target != y_pred, exp_alpha, torch.zeros(1)).sum(1)
            
            s = [BetaInc.apply(c, w, torch.tensor(0.5)) for c, w in zip(correct, wrong)]
            
            if mean:
                return sum(s) / len(y_target)
            else : 
                return sum(s)
        
        elif risk_type == "draw" : 
            
            if post is not None :
                thetas = post
            else:
                thetas = self.rsample()
           
            w_theta = torch.where(y_target != y_pred, thetas, torch.zeros(1)).sum(1)
            
            r = (w_theta >= 0.5).float()
            
            if mean:
                return r.mean()

            return r.sum()
        
        elif risk_type == "expectation" : 

            
            expected_thetas = exp_alpha / exp_alpha.sum(dim=-1, keepdim=True)

            w_theta = torch.where(y_target != y_pred, expected_thetas, torch.zeros(1)).sum(1)

            r = (w_theta >= 0.5).float()
            
            if mean:
                return r.mean()

            return r.sum()

        



    def approximated_risk(self, batch, loss, mean=True,  post=None):

        y_target, y_pred = batch

        thetas = self.rsample()
        if post is not None:
            self.thetas = post
            thetas = self.thetas
            
        else:
            thetas = self.rsample()

        r = loss(y_target, y_pred, thetas).mean(0)

        if mean:
            return r.mean()

        return r.sum()

    def rsample(self):
        self.thetas = Dir(torch.exp(self.alpha)).rsample((self.mc_draws,))
        return self.thetas
    
    def get_theta(self, draw= False):
        if draw :
            #print("drawn")
            return self.rsample()
        if self.thetas is not None:
            #print("here")
            return self.thetas
        else :
            return self.rsample()

    def mean(self):
        return Categorical(self.alpha)

    def mode(self):
        assert all(self.alpha > 1), "can compute mode only of Dirichlet with alpha > 1"

        exp_alpha = torch.exp(self.alpha) - 1

        return exp_alpha / exp_alpha.sum()

    def entropy(self, of_mean=True):

        if of_mean:
            return self.mean().entropy()
        else:
            return Dir(torch.exp(self.alpha)).entropy()
    def __getstate__(self):
        state = self.__dict__.copy()
        
        state.pop('thetas', None)  # when deepcoying the model
        return state


class Categorical():

    def __init__(self, theta, mc_draws=10):
        self.theta = theta
        self.mc_draws = mc_draws
        
    def KL(self, beta):

        t = self.get_theta()

        b = beta / beta.sum()

        return (t * torch.log(t / b)).sum()
    
    def Renyi(self, beta, order):
        # Renyi divergence between two categorical distributions
        t = self.get_theta()
        return (1 / (order - 1)) * torch.log((t ** order / beta ** (order - 1)).sum())
    
    


    def approximated_risk(self, batch, loss, mean=True, post= None):

        t = self.get_theta()

        y_target, y_pred = batch

        r = loss(y_target, y_pred, t)
        
        if mean:
            return r.mean()

        return r.sum()

    def risk(self, batch, risk_type="stochastic", mean=True, post=None):

        t = self.get_theta()

        y_target, y_pred = batch

        w_theta = torch.where(y_target != y_pred, t, torch.zeros(1)).sum(1)

        r = (w_theta >= 0.5).float()

        if mean:
            return r.mean()

        return r.sum()

    def rsample(self):

        t = self.get_theta()

        return t.unsqueeze(0)

    def get_theta(self, draw = False):
        
        return torch.nn.functional.softmax(self.theta, dim=0)

    def entropy(self):

        theta = self.get_theta()
        return - torch.sum(theta * torch.log(theta))
    


class Gaussian():

    def __init__(self, w, mc_draws=10, sigma2=0.5):

        self.w = w #posterior
        self.mc_draws = mc_draws
        self.sigma2 =sigma2
        self.thetas =None
        
    def KL(self, beta):
        # KL divergence between two gaussian distributions
        return torch.sum((self.w - beta) ** 2)* .5/self.sigma2

    def KL_disintegrated(self, beta):

        res = Gaus(self.w, torch.eye(len(self.w)) * self.sigma2).log_prob(self.thetas)
        res -= Gaus(beta, torch.eye(len(self.w)) * self.sigma2).log_prob(self.thetas)
        res = torch.clamp(res.squeeze(0), 0)

        return res
    

    def Renyi(self, beta, order):
        # Renyi divergence between two gaussian distributions

        return order * self.KL(beta)

    
    
    def approximated_risk(self, batch, loss, mean=True, post = None):

        # Risk of the average stochastic classifier, given a certain loss
        y_target, y_pred = batch
        
        if post is not None :
            self.thetas = post
            thetas = self.thetas
        else :
            thetas = self.rsample()

        r = loss(y_target, y_pred, thetas).mean(0)

        if mean:
            return r.mean()

        return r.sum()
    

    
    def risk(self, batch, risk_type="draw", mean=True, post=None):
    
        y_target, y_pred = batch

        if risk_type == "draw" : 
            if post is not None :
                thetas = post
                
            else:
                thetas = self.rsample()
            
            mean_theta = (thetas.sum()/2).detach()
            w_theta = torch.where(y_target != y_pred, thetas, torch.zeros(1)).sum(1)
            
            r = (w_theta >= mean_theta).float()
            
            if mean:
                return r.mean()

            return r.sum()
        
        elif risk_type == "expectation" : 

            
            expected_thetas = torch.reshape(self.w, (-1, 1))
            mean_theta = expected_thetas/2

            w_theta = torch.where(y_target != y_pred, expected_thetas, torch.zeros(1)).sum(1)

            r = (w_theta >= mean_theta).float()
            
            if mean:
                return r.mean()


    
    def rsample(self):
        # Random draw according to the distribution
        #print("rsample")
        #print(self.sigma2)
        w = torch.reshape(self.w, (-1, 1))
        self.thetas = Gaus(w.squeeze(), torch.eye(len(w))*self.sigma2).rsample((self.mc_draws,))
        return self.thetas

    def get_theta(self, draw= False):
        if draw :
            return self.rsample()
        if self.thetas is not None:
            return self.thetas
        else :
            return self.rsample()

    def mean(self):
        return Categorical(self.w)

    def entropy(self, of_mean=True):
        if of_mean:
            return self.mean().entropy()
        else:
            d = len(self.w)
            return 0.5 * d * (1 + torch.log(2*torch.pi*self.sigma2))
        
    def __getstate__(self):
        state = self.__dict__.copy()
        
        state.pop('thetas', None)  # when deepcoying the model
        return state



class Log_Gaussian():

    def __init__(self, w, mc_draws=10, sigma2=0.5):

        self.w = w #posterior
        self.mc_draws = mc_draws
        self.sigma2 =sigma2
        self.thetas =None
        
    def KL(self, beta):
        # KL divergence between two gaussian distributions
        return torch.sum((self.w - beta) ** 2)* .5/self.sigma2

    def KL_disintegrated(self, beta):

        res = LogGaus(self.w, torch.eye(len(self.w)) * self.sigma2).log_prob(self.thetas)
        res -= LogGaus(beta, torch.eye(len(self.w)) * self.sigma2).log_prob(self.thetas)
        res = torch.clamp(res.squeeze(0), 0)

        return res
    

    def Renyi(self, beta, order):
        # Renyi divergence between two gaussian distributions

        return order * self.KL(beta)

    
    
    def approximated_risk(self, batch, loss, mean=True, post = None):

        # Risk of the average stochastic classifier, given a certain loss
        y_target, y_pred = batch
        
        if post is not None :
            self.thetas = post
            thetas = self.thetas
        else :
            thetas = self.rsample()

        r = loss(y_target, y_pred, thetas).mean(0)

        if mean:
            return r.mean()

        return r.sum()
    

    
    def risk(self, batch, risk_type="draw", mean=True, post=None):
    
        y_target, y_pred = batch

        if risk_type == "draw" : 
            if post is not None :
                thetas = post
                
            else:
                thetas = self.rsample()
            
            mean_theta = (thetas.sum()/2).detach()
            w_theta = torch.where(y_target != y_pred, thetas, torch.zeros(1)).sum(1)
            
            r = (w_theta >= mean_theta).float()
            
            if mean:
                return r.mean()

            return r.sum()
        
        elif risk_type == "expectation" : 

            
            expected_thetas = torch.reshape(self.w, (-1, 1))
            mean_theta = expected_thetas/2

            w_theta = torch.where(y_target != y_pred, expected_thetas, torch.zeros(1)).sum(1)

            r = (w_theta >= mean_theta).float()
            
            if mean:
                return r.mean()


    
    def rsample(self):
        # Random draw according to the distribution
        #print("rsample")
        #print(self.sigma2)
        w = torch.reshape(self.w, (-1, 1))
        self.thetas = LogGaus(w.squeeze(), self.sigma2).rsample((self.mc_draws,))
        return self.thetas

    def get_theta(self, draw= False):
        if draw :
            return self.rsample()
        if self.thetas is not None:
            return self.thetas
        else :
            return self.rsample()

    def mean(self):
        return Categorical(self.w)

    def entropy(self, of_mean=True):
        if of_mean:
            return self.mean().entropy()
        else:
            d = len(self.w)
            return 0.5 * d * (1 + torch.log(2*torch.pi*self.sigma2))
        
    def __getstate__(self):
        state = self.__dict__.copy()
        
        state.pop('thetas', None)  # when deepcoying the model
        return state


distr_dict = {
    "dirichlet": Dirichlet,
    "categorical": Categorical,
    "gaussian": Gaussian,
    "log_gaussian": Log_Gaussian,
}
