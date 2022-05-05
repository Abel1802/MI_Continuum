import torch
import torch.nn as nn


class CLUB(nn.Module):
    '''
    '''
    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUB, self).__init__()
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size//2, hidden_size//2),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size//2, hidden_size//2),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size//2, y_dim))

        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                      nn.ReLU(),
                                      nn.Linear(hidden_size//2, hidden_size//2),
                                      nn.ReLU(),
                                      nn.Linear(hidden_size//2, hidden_size//2),
                                      nn.ReLU(),
                                      nn.Linear(hidden_size//2, y_dim),
                                      nn.Tanh())
    
    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar

    def forward(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        sample_size = mu.shape[0]
        random_index = torch.randperm(sample_size).long()
        y_shuffle = y_samples[random_index]
        mu = mu.reshape(-1, mu.shape[-1]) # (bs, y_dim) -> (bs, 1, y_dim) -> (bs, T, y_dim) -> (bs*T, y_dim)
        logvar = logvar.reshape(-1, logvar.shape[-1])
        y_samples = y_samples.reshape(-1, y_samples.shape[-1]) # (bs, T, y_dim) -> (bs*T, y_dim)
        y_shuffle = y_shuffle.reshape(-1, y_shuffle.shape[-1]) # (bs, T, y_dim) -> (bs*T, y_dim)
        
        positive = - (mu - y_samples)**2 / logvar.exp()
        negative = - (mu - y_shuffle)**2 / logvar.exp()
        upper_bound = (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()
        return upper_bound/2.

    def loglikeli(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        mu = mu.reshape(-1, mu.shape[-1]) # (bs, y_dim) -> (bs, 1, y_dim) -> (bs, T, y_dim) -> (bs*T, y_dim)
        logvar = logvar.reshape(-1, logvar.shape[-1])
        y_samples = y_samples.reshape(-1, y_samples.shape[-1]) # (bs, T, y_dim) -> (bs*T, y_dim)
        return (-(mu - y_samples)**2 /logvar.exp()-logvar).sum(dim=1).mean(dim=0)
    
    def learning_loss(self, x_samples, y_samples):
        return -self.loglikeli(x_samples, y_samples)



class MINE(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size):
        super(MINE, self).__init__()
        self.T_func = nn.Sequential(nn.Linear(x_dim + y_dim, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, 1))
    
    def forward(self, x_samples, y_samples):  # samples have shape [sample_size, dim]
        # shuffle and concatenate
        sample_size = y_samples.shape[0]
        random_index = torch.randint(sample_size, (sample_size,)).long()
        y_shuffle = y_samples[random_index]

        x_samples = x_samples.reshape(-1, x_samples.shape[-1]) # (bs, T, x_dim) -> (bs*T, x_dim)
        y_samples = y_samples.reshape(-1, y_samples.shape[-1]) # (bs, T, y_dim) -> (bs*T, y_dim)
        y_shuffle = y_shuffle.reshape(-1, y_shuffle.shape[-1]) # (bs, T, y_dim) -> (bs*T, y_dim)
        

        T0 = self.T_func(torch.cat([x_samples,y_samples], dim = -1))
        T1 = self.T_func(torch.cat([x_samples,y_shuffle], dim = -1))

        lower_bound = T0.mean() - torch.log(T1.exp().mean())

        # compute the negative loss (maximise loss == minimise -loss)
        return lower_bound
    
    def learning_loss(self, x_samples, y_samples):
        return -self.forward(x_samples, y_samples)