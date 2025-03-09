import torch
import torch.nn as nn
import inspect

def _loss_fn(loss):
    if isinstance(loss, str):
        return _str_to_loss(loss)
    elif isinstance(loss, nn.Module):
        return loss
    elif inspect.isclass(loss) and issubclass(loss, nn.Module): # user can send nn.ReLU instead of nn.ReLU()
        return loss()
    else:
        raise ValueError(f"Unsupported loss function: {loss}")


def _str_to_loss(loss):
    loss = loss.lower()
    match loss:
        case 'huber':
            return torch.nn.HuberLoss()
        case 'l1' | 'mae':
            return torch.nn.L1Loss()
        case 'l2' | 'mse':
            return torch.nn.MSELoss()
        case 'smooth_l1':
            return torch.nn.SmoothL1Loss()
        case 'soft_margin':
            return torch.nn.SoftMarginLoss()
        case 'nll':
            return torch.nn.NLLLoss()
        case 'cross_entropy':
            return torch.nn.CrossEntropyLoss()
        case 'binary_cross_entropy', 'bce':
            return torch.nn.BCELoss()
        case 'binary_cross_entropy_with_logits', 'bce_logits':
            return torch.nn.BCEWithLogitsLoss()
        case 'poisson_nll':
            return torch.nn.PoissonNLLLoss()
        case 'kl_div':
            return torch.nn.KLDivLoss()
        case 'cosine_similarity':
            return torch.nn.CosineSimilarity()
        case 'cosine_embedding_loss':
            return torch.nn.CosineEmbeddingLoss()
        case 'hinge_embedding_loss':
            return torch.nn.HingeEmbeddingLoss()
        case 'multi_margin_loss':
            return torch.nn.MultiMarginLoss()
        case 'multi_label_margin_loss':
            return torch.nn.MultiLabelMarginLoss()
        case 'multi_label_soft_margin_loss':
            return torch.nn.MultiLabelSoftMarginLoss()
        case 'multi_label_cross_entropy':
            return torch.nn.MultiLabelCrossEntropyLoss()
        case 'triplet_margin_loss':
            return torch.nn.TripletMarginLoss()
        case 'triplet_margin_with_distance_loss':
            return torch.nn.TripletMarginWithDistanceLoss()
        case 'cosine_embedding_loss':
            return torch.nn.CosineEmbeddingLoss()
        case 'margin_ranking_loss':
            return torch.nn.MarginRankingLoss()
        case 'hinge_loss':
            return torch.nn.HingeLoss()
        case 'cosine_similarity':
            return torch.nn.CosineSimilarity()
        case 'cosine_embedding_loss':
            return torch.nn.CosineEmbeddingLoss()
        case 'margin_ranking_loss':
            return torch.nn.MarginRankingLoss()

    raise ValueError(f"Unsupported loss function: {loss}")
