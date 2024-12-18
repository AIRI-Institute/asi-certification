import torch
import numpy as np
# import kornia as K
from math import ceil, pi
from scipy.stats import norm
from statsmodels.stats.proportion import proportion_confint
import time

# import matplotlib.pyplot as plt
# from IPython.display import clear_output

class Smooth(object):
    """A smoothed classifier g """
    # to abstain, Smooth returns this int
    ABSTAIN = -1.0
        
    def __init__(self, base_model: torch.nn.Module, device: str, num_classes: int, sigma: float, alpha: float, mode: str, normalize1, emb_size=512):
        self.base_model = base_model
        self.num_classes = num_classes
        self.sigma = sigma
        self.device = device
        self.transform_mode = mode
        self.alpha = alpha
        self.L = np.sqrt(2 / np.pi / sigma ** 2)
        self.normalize1 = normalize1
        self.emb_size = emb_size


        
    def embedding_risk_lcb(self, x: torch.tensor, true_centroid: torch.tensor, adv_centroid: torch.tensor):
        n_samples = self.K * self.N
        batch_size = 100
        alpha = self.alpha

        get_mean_quadr = lambda a, b: torch.mean(a, axis=0) @ torch.mean(torch.transpose(b, 1, 0), axis=1)
        get_mean_lin = lambda a, c: a @ torch.transpose(c.unsqueeze(0), 1, 0)
        
        with torch.no_grad():
            new_embeddings = self._sample_smoothed(x, 1, n_samples, batch_size)[0]

            # centr_dist = 2 * torch.sum((true_centroid - adv_centroid) ** 2)
            centr_dist = 2 * torch.sqrt(torch.sum((true_centroid - adv_centroid) ** 2))

            new_embedding = torch.mean(new_embeddings, dim=0)
            # print("AAA",torch.norm(new_embedding),torch.norm(true_centroid), torch.norm(adv_centroid))
            lin = get_mean_lin(new_embedding, true_centroid - adv_centroid)
            # print(new_embedding.shape, (true_centroid - adv_centroid).shape)
            # print(lin)
            conf_int = self._confidence_intervals(lin, n_samples, alpha, 4.)
            # print(conf_int)
            lcb_gamma = conf_int[0] / centr_dist
    

        return lcb_gamma

            
    def certified_radius(self, x: torch.tensor, true_centroid: torch.tensor, adv_centroid: torch.tensor, g_x=None):
        """Improved certified radius in l2-norm for sample x"""
        
        """
        :return: lower estimate of l2-norm of perturbation that doesn't change prediction on x
        """
        n_samples = self.K * self.N
        batch_size = 100
        alpha = self.alpha
        
        lcb_gamma = self.embedding_risk_lcb(x, true_centroid, adv_centroid) + 0.5
        # print("lcb_gamma:", lcb_gamma, "ppf(lcb_gamma):", norm.ppf(lcb_gamma.cpu()))
        if len(true_centroid.shape) == 1:
            delta = self._adversarial_emb(g_x, true_centroid, adv_centroid)
            radius_as_in_article =  abs(delta * self.sigma * np.sqrt(pi / 2))
        else:
            deltas = self._adversarial_emb_batch(g_x, true_centroid, adv_centroid)
            radius_as_in_article =  abs(deltas * self.sigma * np.sqrt(pi / 2))

        return lcb_gamma, self.sigma * (norm.ppf(lcb_gamma.cpu())), radius_as_in_article

        
    
    def predict(self, args, x: torch.tensor, centroids: torch.tensor, centroid_classes: torch.tensor):
        """Define predicted by smoothed model class and adversarial class on sample x
        """
        """
        :return: (predicted class, adversarial class, predicted class centroid, adversarial class centroid, n_samples) or ABSTAIN
        """
        
        alpha = args.alpha
        k_repeats = args.K
        n_samples = args.N
        
        self.K = args.K
        self.N = args.N
        
        batch_size = args.batch
        
        get_mean_quadr = lambda a, b: torch.mean(a, axis=0) @ torch.mean(torch.transpose(b, 1, 0), axis=1)
        get_mean_lin = lambda a: 2 * a @ torch.transpose(centroids, 1, 0)
        
        with torch.no_grad():
            
            mean_quadr = 0
            mean_lin = torch.zeros(self.num_classes, device=self.device)
            classes_const = torch.diag(centroids @ torch.transpose(centroids, 1, 0), 0)

            for k in range(k_repeats):
                # sample 2n values of f(x + eps) for each of m realization
                new_embeddings = self._sample_smoothed(x, 1, 2 * n_samples, batch_size)[0]
                #for each class, update coef * (<sumf(x+eps_i), sumf(x+eps_j)> and <sumf(x+eps_i)- c_k>
                
                #calculate conf bounds for mean <f(x+eps_i), f(x+eps_j)>
                new_mean_quadr = get_mean_quadr(new_embeddings[:n_samples, :], new_embeddings[n_samples:2*n_samples, :])
                #calculate conf bounds for <f(x+eps_i), c_k>

                # norm = torch.norm(torch.mean(new_embeddings, dim=0), p=2, dim=-1)
                
                new_mean_lin = get_mean_lin(torch.mean(new_embeddings, dim=0)) # [num_classes]
                
                mean_quadr = (k * mean_quadr + new_mean_quadr) / (k + 1)

                # print("Mean Lin", mean_lin.shape, new_mean_lin.shape)
                mean_lin = (k * mean_lin + new_mean_lin) / (k + 1)
                    

                #confidence intervals for means of s_k for all classes
                conf_ints_quadr = self._confidence_intervals(mean_quadr, (n_samples ** 2) * (k + 1), alpha, 4.)
                conf_ints_lin = self._confidence_intervals(mean_lin, n_samples * (k + 1), alpha, 4.)
                
                index = torch.LongTensor([1, 0])
                
                conf_ints = torch.sqrt(conf_ints_quadr - conf_ints_lin[index] + classes_const)


                if self._robustness_condition(conf_ints):
                    idxes = torch.argsort(conf_ints[1, ...])[:2].to(centroid_classes.device)
                    return centroid_classes[idxes], centroids[idxes], n_samples * (k+1)
            
            return Smooth.ABSTAIN
    
    
    def predict_without_abstain(self, args, x: torch.tensor, centroids: torch.tensor, centroid_classes: torch.tensor):
        """Define predicted by smoothed model class and adversarial class on sample x
        """
        """
        :return: (predicted class, adversarial class, predicted class centroid, adversarial class centroid, n_samples) or ABSTAIN
        """
        
        alpha = args.alpha
        k_repeats = args.K
        n_samples = args.N
        
        self.K = args.K
        self.N = args.N
        
        batch_size = args.batch
        
        get_mean_quadr = lambda a, b: torch.mean(a, axis=0) @ torch.mean(torch.transpose(b, 1, 0), axis=1)
        get_mean_lin = lambda a: 2 * a @ torch.transpose(centroids, 1, 0)
        
        with torch.no_grad():
            
            mean_quadr = 0
            mean_lin = torch.zeros(self.num_classes, device=self.device)
            classes_const = torch.diag(centroids @ torch.transpose(centroids, 1, 0), 0)
            
            for k in range(k_repeats):
                # sample 2n values of f(x + eps) for each of m realization
                new_embeddings = self._sample_smoothed(x, 1, 2 * n_samples, batch_size)[0]
                #for each class, update coef * (<sumf(x+eps_i), sumf(x+eps_j)> and <sumf(x+eps_i)- c_k>
                
                #calculate conf bounds for mean <f(x+eps_i), f(x+eps_j)>
                new_mean_quadr = get_mean_quadr(new_embeddings[:n_samples, :], new_embeddings[n_samples:2*n_samples, :])
                #calculate conf bounds for <f(x+eps_i), c_k>
                # norm = torch.norm(torch.mean(new_embeddings, dim=0), p=2, dim=-1)
                
                new_mean_lin = get_mean_lin(torch.mean(new_embeddings, dim=0)) # [num_classes]
                
                mean_quadr = (k * mean_quadr + new_mean_quadr) / (k + 1)
                
                # print("Mean Lin", mean_lin.shape, new_mean_lin.shape)
                mean_lin = (k * mean_lin + new_mean_lin) / (k + 1)
                
                #confidence intervals for means of s_k for all classes
                conf_ints_quadr = self._confidence_intervals(mean_quadr, (n_samples ** 2) * (k + 1), alpha, 4.)
                conf_ints_lin = self._confidence_intervals(mean_lin, n_samples * (k + 1), alpha, 4.)
                
                index = torch.LongTensor([1, 0])
                
                conf_ints = torch.sqrt(conf_ints_quadr - conf_ints_lin[index] + classes_const)
                
                if self._robustness_condition(conf_ints):
                    idxes = torch.argsort(conf_ints[1, ...])[:2].to(centroid_classes.device)
                    return centroid_classes[idxes], centroids[idxes], n_samples * (k+1)
            
            idxes = torch.argsort(conf_ints[1, ...])[:2].to(centroid_classes.device)
            return centroid_classes[idxes], centroids[idxes], n_samples * (k+1)
    
    
    def _sample_smoothed(self, x: torch.tensor, m_values: int, n_samples: int, batch_size: int):
        """Sample values of g(x) Monte-Carlo estimation (without normalization) // actually with"""
        
        """
        :return: tensor of size [m_values, emb_size]
        """
        
        num = m_values * n_samples
        embeddings = torch.empty((1, self.emb_size)).to(self.device)
        
        with torch.no_grad():
            for _ in range(ceil(num / batch_size)):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size

                # batch = x.repeat((this_batch_size, 1, 1, 1))
                # X shape is [T]
                batch = x.repeat((this_batch_size, 1))
                # print(batch.shape)
                if self.transform_mode == 'small-norm':
                    noise = torch.randn_like(batch, device=self.device) * self.sigma
                    qq  = batch + noise
                    # if self.base_model.model_name == "wavlm":
                    #     qq  = (batch + noise).reshape(this_batch_size, 1, -1)
                    # elif self.base_model.model_name == "ecapa-tdnn":
                    #     qq  = (batch + noise).reshape(this_batch_size, -1)
                    # else:
                    #     qq  = (batch + noise).squeeze(1)
                    # qq  = (batch + noise).squeeze(1).squeeze(1)
                    # print(qq.shape)
                    batch_emb = self.base_model(qq).reshape(this_batch_size, self.emb_size)

                    # print("batch_emb.shape", batch_emb.shape)
  
                elif self.transform_mode == 'gamma':
                    gammas = torch.exp(self.sigma * torch.randn(batch.shape[0], device=self.device))[:, None, None, None]
                    batch_emb = self.base_model(torch.pow(batch, gammas)).to(self.device)
                elif self.transform_mode == 'translate':
                    img_size = batch.shape[-1]
                    translation = torch.randn((batch.shape[0], 2), device=self.device) * self.sigma * img_size
                    translated = k_transform.translate(batch, translation=translation, padding_mode='reflection')
                    batch_emb = self.base_model(translated).to(self.device)
                # print(embeddings.shape, batch_emb.shape)
                if self.normalize1:
                    norms = torch.norm(batch_emb, p=2, dim=-1, keepdim=True)
                    batch_emb /= norms
                # print(embeddings.shape, batch_emb.shape)
                batch_emb = batch_emb.reshape(this_batch_size, -1)
                embeddings = torch.cat((embeddings, batch_emb))

            embeddings = embeddings[1:].reshape(m_values, n_samples, -1)
            
            
        return embeddings
    
    
    
    def _robustness_condition(self, conf_ints: torch.tensor):
        """
        Check if confidence intervals for the three smallest means don't intersect
        It means, the closest class always predicted with 1-alpha probability and there is an adversarial class
        """

        i_pred = torch.argmin(conf_ints[0, ...]).cpu()
        diff1 = conf_ints[1, i_pred] - conf_ints[0, np.r_[:i_pred, i_pred+1:self.num_classes]] # b_min - a_i, i != min
        cond1 =  torch.all(diff1 <= 0)
        
        ci_wo_pred = conf_ints[0, np.r_[:i_pred, i_pred+1:self.num_classes]]
        
        i_adv = torch.argmin(ci_wo_pred).cpu()
        i_adv = i_adv + (i_adv >= i_pred)
        diff2 = conf_ints[1, i_adv] - conf_ints[0, np.r_[:min(i_pred, i_adv), min(i_pred, i_adv) + 1:max(i_pred, i_adv), max(i_pred, i_adv) + 1:self.num_classes]]
        cond2 = torch.all(diff2 <= 0)
        
        return cond1 & cond2

    
    def _adversarial_emb(self, z: torch.tensor, x: torch.tensor, y: torch.tensor):
        # print("i m on non batch")
        return (0.5 * (torch.norm(y, p=2) ** 2 - torch.norm(x, p=2) ** 2) -  z @ (y - x)) / torch.norm(y - x, p=2) ** 2
    
    
    
    def _adversarial_emb_batch(self, z: torch.tensor, x: torch.tensor, y: torch.tensor):
        def norm_batch(a):
            return torch.diag(a @ torch.transpose(a, 0, 1))
        
        return (0.5 * (norm_batch(y) - norm_batch(x)) - torch.diag(z @ torch.transpose(y - x, 0, 1))) / norm_batch(y - x) 
               
        
    def _confidence_intervals(self, x_means: torch.tensor, n: int, alpha: float, bound_const: float):
        """
        Confidence interval on sample's mean using Hoeffding's inequality 
        """
        t = np.sqrt((- np.log(alpha / 2) / (((bound_const ** 2) / 2) * n)))
        square_conf_ints = x_means.repeat(2, 1) + torch.tensor([[-t], [+t]]).to(self.device)
        
        return square_conf_ints
    
    def _lower_confidence_bound(self, NA: int, N: int, alpha: float) -> float:
        """ Returns a (1 - alpha) lower confidence bound on a bernoulli proportion.
        This function uses the Clopper-Pearson method.
        :param NA: the number of "successes"
        :param N: the number of total draws
        :param alpha: the confidence level
        :return: a lower bound on the binomial proportion which holds true w.p at least (1 - alpha) over the samples
        """
        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]
    

def divide_batch(batch, target, n_support):
    
    def supp_idxs(c):
        return torch.where(target==c)[0][:n_support]
    
    classes = torch.unique(target)
    n_classes = len(classes)
    
    n_query = len(torch.where(target==classes[0].item())[0]) - n_support
    support_idxs = list(map(supp_idxs, classes))
    support_samples = torch.vstack([batch[idx_list] for idx_list in support_idxs])
    support_target = torch.hstack([target[idx_list] for idx_list in support_idxs])
    
    query_idxs = torch.stack(list(map(lambda c: torch.where(target==c)[0][n_support:], classes))).view(-1)
    
    query_samples = batch[query_idxs]
    query_target = target[query_idxs]
    
    return support_samples, support_target, query_samples, query_target

def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)





