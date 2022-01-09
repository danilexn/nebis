import os
import logging

from torch.utils.tensorboard import SummaryWriter
import torch
from torch.nn import DataParallel

from nebis.data import get_datareader
from nebis.models import get_model, parallel_predict
from nebis.utils.args import argument_parser
from nebis.utils import set_seed
from nebis.utils.evaluate import get_evaluator

from transformers import BertModel


def hook_step(step, values):
    for v in values:
        writer.add_scalar(
            v, values[v], step,
        )


if __name__ == "__main__":
    # Argument parser load and customisation
    parser = argument_parser()
    args = parser.parse_args()
    set_seed(args)

    # TensorBoard writer
    logging.info("Configuring SummaryWriter at {}".format(args.log_dir))
    writer = SummaryWriter(args.log_dir)

    # Load BERT config
    logging.info("Loading DNABERT model from {}".format(args.pretrained_bert_in))
    pretrained_bert = BertModel.from_pretrained(args.pretrained_bert_in)
    args.bert_config = pretrained_bert.config

    if args.model_in is not None:
        if os.path.isdir(args.model_in):
            args.model_in = os.path.join(args.model_in, "model.pth")
            if not os.path.exists(args.model_in):
                raise FileNotFoundError(
                    "The specified path {} does not contain a pytorch model".format(
                        args.model_in
                    )
                )

    # Load model
    logging.info("Loading '{}' model from {}".format(args.model, args.model_in))
    model = torch.load(args.model_in)
    model.to(args.device)

    # Create model
    logging.info("Creating '{}' model".format(args.model))
    model = get_model(args.model)(args, BERT=pretrained_bert)
    if args.n_gpu > 1:
        logging.info("Parallelising model over {} GPUs".format(args.n_gpu))
        model = DataParallel(model)

    model.to(args.device)

    logging.info("Reading dataset")
    dataset = get_datareader("{}_{}".format(args.model, args.downstream))(args)
    dataset.load(args.data_dir)

    logging.info("Running inference")
    batch_size = args.single_batch * args.n_gpu
    if not isinstance(model, DataParallel):
        Ys, Ps, Hs = model.predict(dataset.inference(batch_size=batch_size),)
    else:
        Ys, Ps, Hs = parallel_predict(model, dataset.inference(batch_size=batch_size),)

    evaluator = get_evaluator(args.downstream)(Ys, Ps)
    evaluation = evaluator.evaluate()
