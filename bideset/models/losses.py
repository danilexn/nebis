import torch
import torch.nn.functional as F

def MTLR_survival_loss(y_pred, y_true, E, tri_matrix, reduction="mean"):
    """
    Compute the MTLR survival loss
    """
    # Get censored index and uncensored index
    censor_idx = []
    uncensor_idx = []
    for i in range(len(E)):
        # If this is a uncensored data point
        if E[i] == 1:
            # Add to uncensored index list
            uncensor_idx.append(i)
        else:
            # Add to censored index list
            censor_idx.append(i)

    # Separate y_true and y_pred
    y_pred_censor = y_pred[censor_idx]
    y_true_censor = y_true[censor_idx]
    y_pred_uncensor = y_pred[uncensor_idx]
    y_true_uncensor = y_true[uncensor_idx]

    # Calculate likelihood for censored datapoint
    phi_censor = torch.exp(torch.mm(y_pred_censor, tri_matrix))
    reduc_phi_censor = torch.sum(phi_censor * y_true_censor, dim=1)

    # Calculate likelihood for uncensored datapoint
    phi_uncensor = torch.exp(torch.mm(y_pred_uncensor, tri_matrix))
    reduc_phi_uncensor = torch.sum(phi_uncensor * y_true_uncensor, dim=1)

    # Likelihood normalisation
    z_censor = torch.exp(torch.mm(y_pred_censor, tri_matrix))
    reduc_z_censor = torch.sum(z_censor, dim=1)
    z_uncensor = torch.exp(torch.mm(y_pred_uncensor, tri_matrix))
    reduc_z_uncensor = torch.sum(z_uncensor, dim=1)

    # MTLR loss
    loss = -(
        torch.sum(torch.log(reduc_phi_censor))
        + torch.sum(torch.log(reduc_phi_uncensor))
        - torch.sum(torch.log(reduc_z_censor))
        - torch.sum(torch.log(reduc_z_uncensor))
    )

    if reduction == "mean":
        loss = loss / E.shape[0]

    return loss


def cross_entropy_loss(logits, y, weight):
    return F.cross_entropy(
                    logits,
                    y,
                    weight=weight,
                )