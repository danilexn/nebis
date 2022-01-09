import os
import logging

from torch.utils.tensorboard import SummaryWriter
import torch.load as load

from nebis.data import get_datareader
from nebis.models import get_model
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

    # Load model
    logging.info("Loading '{}' model from {}".format(args.model, args.model_in))
    model = load(args.model_in)
    model.to(args.device)

    dataset = get_datareader("{}_{}".format(args.model, args.downstream))(args)
    dataset.load()

    batch_size = args.single_batch
    Ys, Ps, Hs = model.predict(dataset.predicting())

    evaluator = get_evaluator(args.downstream)(Ys, Ps)
    evaluation = evaluator.evaluate()
