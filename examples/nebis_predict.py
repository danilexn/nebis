import os

from torch.utils.tensorboard import SummaryWriter
from torch.nn import DataParallel

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
    writer = SummaryWriter(args.log_dir)

    # Load BERT config
    pretrained_bert = BertModel.from_pretrained(args.pretrained_bert_in)
    args.bert_config = pretrained_bert.config

    # Create model
    model = get_model(args.model)(args, BERT=pretrained_bert)
    if args.n_gpu >= 1:
        model = DataParallel(model)

    model.to(args.device)

    dataset = get_datareader("{}_{}".format(args.model, args.downstream))(args)
    dataset.load()

    batch_size = args.single_batch * args.n_gpu
    if not isinstance(model, DataParallel):
        Ys, Ps, Hs = model.predict(dataset.predicting())
    else:
        Ys, Ps, Hs = model.module.predict(dataset.predicting())

    evaluator = get_evaluator(args.downstream)(Ys, Ps)
    evaluation = evaluator.evaluate()
