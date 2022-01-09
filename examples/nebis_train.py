import os
import logging

from torch.utils.tensorboard import SummaryWriter
from torch.nn import DataParallel

from nebis.data import get_datareader
from nebis.models import get_model, parallel_fit, profiled_parallel_fit
from nebis.utils.args import argument_parser
from nebis.utils import set_seed

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

    # Create model
    logging.info("Creating '{}' model".format(args.model))
    model = get_model(args.model)(args, BERT=pretrained_bert)
    if args.n_gpu > 1:
        logging.info("Parallelising model over {} GPUs".format(args.n_gpu))
        model = DataParallel(model)

    model.to(args.device)

    logging.info("Reading dataset")
    dataset = get_datareader("{}_{}".format(args.model, args.downstream))(args)
    dataset.load()

    logging.info("Training model")
    batch_size = args.single_batch * args.n_gpu
    if not isinstance(model, DataParallel):
        model.fit(
            dataset.fitting(batch_size=batch_size),
            dataset.predicting(batch_size=batch_size),
            args,
            hook_step=hook_step,
        )
    else:
        parallel_fit(
            model,
            dataset.fitting(batch_size=batch_size),
            dataset.predicting(batch_size=batch_size),
            args,
            hook_step=hook_step,
        )

    logging.info("Saving trained model at {}".format(args.model_out))
    model_path = os.path.join(args.model_out, "model.pth")
    if not isinstance(model, DataParallel):
        model.save(model_path)
    else:
        model.module.save(model_path)
