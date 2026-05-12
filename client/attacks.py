from client.client_utils import estimate_static_RLU, estimated_entropy_from_grad, estimate_static_RLU_with_posterior
from client.client_utils import estimate_static_LLG,estimate_static_ZLG
from client.client_utils import create_synthetic_basis_matrix, calculate_distribution_ratios, calculate_dynamic_boost
import numpy as np
import torch
import copy
import scipy



def VLI_attack( gradients_for_prediction, args): 
    #VLI
    alpha= 1.7 #1.0 #1.0 #1.5
    beta = 2.0 #1.0 #2.0 #3.0
    negative_elements = gradients_for_prediction[gradients_for_prediction < 0]
    
    if len(negative_elements) > 0:
        sum_negative = torch.sum(negative_elements)
    else:
        sum_negative = torch.sum(gradients_for_prediction)

    base_diagonal_val = sum_negative / (args.batch_size* args.local_epochs) 
    
    
    Basis = create_synthetic_basis_matrix(args.n_classes, base_diagonal_val)
    probs = calculate_distribution_ratios(gradients_for_prediction, Basis)
    max_p = np.max(probs)
    length = np.linalg.norm(gradients_for_prediction)


    
    
    
    boost_factor = calculate_dynamic_boost(probs, length)
    
    final_diagonal_val = base_diagonal_val * boost_factor
    Basis = create_synthetic_basis_matrix(args.n_classes, final_diagonal_val)

    residual = gradients_for_prediction.clone()
    counts = np.zeros(args.n_classes, dtype=int)
    
    for step in range(args.batch_size * args.local_epochs):
        scores = np.dot(residual, Basis)
        best_idx = np.argmax(scores)
        counts[best_idx] += 1
        
        projection_val = 1.0
        component_to_remove =   Basis[:, best_idx]
        residual = residual - component_to_remove



    return counts

def ZLGp_attack(model, gradients_for_prediction, O_bar, pj, aux_dataset, args):
    new_O_bar, new_pj = estimate_static_ZLG(args, copy.deepcopy(model), aux_dataset)
    new_O_bar = (new_O_bar + O_bar) / 2
    new_pj = (new_pj + pj) / 2


    n = []
    for i in range(args.n_classes):
        nj = args.batch_size * args.local_epochs * (new_pj[i].detach().cpu() - gradients_for_prediction[i] / new_O_bar.detach().cpu())
        n.append(max(int(nj.item()), 0))
    prop = (args.local_epochs * args.batch_size) / sum(n)
    for i in range(args.n_classes):
        n[i] = round(n[i] * prop)
    return n



def LLG_attack(model, gradients_for_prediction, args):
    

    h1_extraction = []
    negative_gradient = 0.0
    for i_cg, class_gradient in enumerate(gradients_for_prediction):
        if class_gradient < 0:
            h1_extraction.append((i_cg, class_gradient))
            negative_gradient += class_gradient
    
    new_impact = (1+1/args.n_classes)* (negative_gradient/ (args.batch_size))
    prediction = []

    for (i_c, _) in h1_extraction:
        prediction.append(i_c)
        gradients_for_prediction[i_c] = gradients_for_prediction[i_c].add(-new_impact)

    for _ in range(args.batch_size - len(prediction)):
        # add minimal candidate, likely to be doubled, to prediction
        min_id = torch.argmin(gradients_for_prediction).item()
        prediction.append(min_id)

        # add the mean value of one occurrence to the candidate
        gradients_for_prediction[min_id] = gradients_for_prediction[min_id].add(-new_impact)

    n = []
    for i in range(args.n_classes):
        n.append(args.local_epochs * prediction.count(i))
    return n


def LLGp_attack(model, gradients_for_prediction, impact, offset, aux_dataset, args):
    new_impact, new_offset = estimate_static_LLG(args, copy.deepcopy(model), aux_dataset)
    new_impact = (new_impact + impact) / 2
    new_offset = (new_offset + offset) / 2

    h1_extraction = []

    for i_cg, class_gradient in enumerate(gradients_for_prediction):
        if class_gradient < 0:
            h1_extraction.append((i_cg, class_gradient))

    gradients_for_prediction -= new_offset
    prediction = []

    for (i_c, _) in h1_extraction:
        prediction.append(i_c)
        gradients_for_prediction[i_c] = gradients_for_prediction[i_c].add(-impact)

    for _ in range(args.batch_size - len(prediction)):
        # add minimal candidate, likely to be doubled, to prediction
        min_id = torch.argmin(gradients_for_prediction).item()
        prediction.append(min_id)

        # add the mean value of one occurrence to the candidate
        gradients_for_prediction[min_id] = gradients_for_prediction[min_id].add(-new_impact)

    n = []
    for i in range(args.n_classes):
        n.append(args.local_epochs * prediction.count(i))
    return n
 

def RLU_attack(model,w_grad_epochs, b_grad_epochs , latent_dim, mu, aux_dataset, args):
    new_mu, new_shift = estimate_static_RLU(args, model, aux_dataset)
    new_shift = scipy.special.softmax(new_mu)
    O = torch.zeros(latent_dim)
    
    for d in range(latent_dim):
        O[d] = torch.mean(w_grad_epochs[:, d] / b_grad_epochs)
    
    rho = np.zeros(args.local_epochs)
    gamma = args.gamma
    for t in range(args.local_epochs):
        rho[t] = (1 - gamma ** (args.local_epochs - t)) / (1 - gamma)
    rho_mean = np.sum(rho) / args.local_epochs
    n = estimated_entropy_from_grad(args, new_shift*rho_mean, b_grad_epochs.detach().cpu().tolist(),
                                    args.batch_size * args.local_epochs)
    new_shift_softmax = estimate_static_RLU_with_posterior(args, n, mu, new_mu, O)
    n = estimated_entropy_from_grad(args, new_shift_softmax*rho_mean,b_grad_epochs.detach().cpu().tolist(), args.batch_size*args.local_epochs)
    return n