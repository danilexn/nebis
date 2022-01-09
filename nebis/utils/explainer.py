import os
import gc
import logging
from tqdm import tqdm

import torch

from nebis.data import get_datareader
from nebis.utils.args import argument_parser
from nebis.utils import set_seed

from captum.attr import IntegratedGradients, DeepLiftShap, DeepLift, InputXGradient, LRP

M_INTEGRATEDGRADIENT = "intgrad"
M_INPUTXGRAD = "inputxgrad"
M_DEEPLIFT = "deeplift"
M_DEEPSHAP = "deepshap"
M_LRP = "relevance"

_attr_algorithm = {
    M_INTEGRATEDGRADIENT: IntegratedGradients,
    M_INPUTXGRAD: InputXGradient,
    M_DEEPLIFT: DeepLift,
    M_DEEPSHAP: DeepLiftShap,
    M_LRP: LRP,
}


def calc_attributions(
    explainer, _model, input_data, input_label, actual_label, args, **kwargs
):
    exp = explainer(_model)
    pred_class = actual_label
    act_class = input_label

    _embed_shape_0 = input_data[0].shape

    diffs_g = []
    attrs_g = []

    for j in [act_class, pred_class]:
        # TODO: adjust the seed as parameter
        set_seed(args)

        _zeros_0 = torch.ones(_embed_shape_0).to(args.device)

        if explainer == IntegratedGradients:
            attributions_g = exp.attribute(
                input_data[0],
                _zeros_0,
                target=j,
                additional_forward_args=(input_data[1], input_data[2]),
                **kwargs
            )
        elif explainer == DeepLiftShap:
            attributions_g = exp.attribute(
                input_data[0],
                _zeros_0,
                target=j,
                additional_forward_args=(input_data[1], input_data[2]),
            )
        else:
            attributions_g = exp.attribute(
                input_data[0],
                target=j,
                additional_forward_args=(input_data[1], input_data[2]),
            )
        diffs_g.append(attributions_g.sum() / torch.norm(attributions_g))
        attributions_g = attributions_g.sum(dim=2) / torch.norm(attributions_g)
        attrs_g.append(attributions_g)

    return attrs_g, diffs_g


if __name__ == "__main__":
    # Argument parser load and customisation
    parser = argument_parser()
    args = parser.parse_args()

    logging.info("Reading dataset")
    dataset = get_datareader("{}_{}".format(args.model, args.downstream))(args)
    dataset.load()

    logging.info("Loading '{}' model from {}".format(args.model, args.model_in))
    model = torch.load(args.model_in)
    model.to(args.device)
    model.eval()

    logging.debug("#Computing {} with {} steps".format(args.attr_steps))

    dataloader_attr = dataset.predicting()
    pbar = tqdm(dataloader_attr, position=0, leave=False)

    for batch in pbar:
        # Move targets to device
        if torch.is_tensor(batch[2]):
            target = batch[2].to(args.device)
        else:
            target = [t.to(args.device) for t in batch[2]]

        # Move features to device
        batch = tuple(t.to(args.device) for t in batch[0:2])

        inputs = {"X_mutome": batch[0], "X_omics": batch[1]}

        with torch.no_grad():
            Y, H = model.forward(**inputs)

        Y_pred = model.Downstream.prediction(Y.detach())
        target = [t.detach().cpu().numpy() for t in target]

        _input = {
            "input_ids": _dataset.tensors[0][i].to(DEVICE),
            "omic_input": _dataset.tensors[4][i].to(DEVICE),
        }

        _input = map2layer(model, **_input)

        # del model
        gc.collect()
        torch.cuda.empty_cache()

        INTERNAL_BATCH = (
            2
            if (args.attribution in [M_DEEPLIFT, M_DEEPSHAP])
            else args.attr_internal_batch
        )

        attrs_g, diffs_g = calc_attributions(
            _attr_algorithm[args.attribution],
            model,
            _input,
            _label,
            p_class,
            args,
            n_steps=args.attr_steps,
            internal_batch_size=INTERNAL_BATCH,
        )

        gc.collect()
        torch.cuda.empty_cache()

    logging.debug("#Execution completed!")
