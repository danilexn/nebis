import os

from torch.utils.tensorboard import SummaryWriter
from torch.nn import DataParallel

from nebis.data import get_dataset
from nebis.models import get_model
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
    writer = SummaryWriter(args.log_dir)

    # Load BERT config
    pretrained_bert = BertModel.from_pretrained(args.pretrained_bert_in)
    args.bert_config = pretrained_bert.config

    # Create model
    model = get_model(args.model)(args, BERT=pretrained_bert)
    if args.n_gpu >= 1:
        model = DataParallel(model)

    model.to(args.device)

    dataset = get_dataset("{}_{}".format(args.model, args.downstream))(args)
    dataset.load()

    batch_size = args.single_batch * args.n_gpu
    if not isinstance(model, DataParallel):
        model.fit(
            dataset.fitting(batch_size=batch_size),
            dataset.predicting(),
            args,
            hook_step=hook_step,
        )
    else:
        model.module.fit(
            dataset.fitting(batch_size=batch_size),
            dataset.predicting(),
            args,
            hook_step=hook_step,
        )

    model_path = os.path.join(args.model_out, "model.pth")
    if not isinstance(model, DataParallel):
        model.save(model_path)
    else:
        model.module.save(model_path)
