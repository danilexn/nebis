import os
import logging
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter
from torch.nn import DataParallel

from nebis.data import get_datareader
from nebis.utils.parallel import ListDataParallel
from nebis.models import get_model, parallel_fit, profiled_parallel_fit
from nebis.utils.args import argument_parser
from nebis.utils import set_seed

from transformers import BertModel

DATE = datetime.today().strftime("%Y%m%d%H%M%S")


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

    tb_dir = os.path.join(
        args.log_dir,
        "{}_{}_{}_{}_{}_{}_{}_{}".format(
            args.model,
            args.downstream,
            DATE,
            args.digitize_bins,
            args.p_dropout,
            args.learning_rate,
            args.pooling_sequence,
            args.activation,
        ),
    )

    if not os.path.exists(args.log_dir):
        logging.debug("#Creating logging directory at {}".format(args.log_dir))
        os.makedirs(args.log_dir)

    logging.basicConfig(
        format="%(asctime)s:%(levelname)s:%(message)s",
        filename=os.path.join(
            args.log_dir,
            "{}_{}_{}_{}_{}_{}_{}_{}.log".format(
                args.model,
                args.downstream,
                DATE,
                args.digitize_bins,
                args.p_dropout,
                args.learning_rate,
                args.pooling_sequence,
                args.activation,
            ),
        ),
        level=logging.DEBUG,
    )

    if not os.path.exists(args.log_dir):
        logging.debug("#Creating logging directory at {}".format(args.log_dir))
        os.makedirs(args.log_dir)

    if args.profile_dir is not None and not os.path.exists(args.profile_dir):
        logging.debug("#Creating profiling directory at {}".format(args.model_out))
        os.makedirs(args.profile_dir)

    if not os.path.exists(tb_dir):
        logging.debug("#Creating TensorBoard directory at {}".format(tb_dir))
        os.makedirs(tb_dir)

    # TensorBoard writer
    logging.info("Configuring SummaryWriter at {}".format(tb_dir))
    writer = SummaryWriter(tb_dir)

    if not os.path.exists(args.model_out):
        logging.debug("#Creating output directory at {}".format(args.model_out))
        os.makedirs(args.model_out)

    # Load BERT config
    logging.info("Loading DNABERT model from {}".format(args.pretrained_bert_in))
    pretrained_bert = BertModel.from_pretrained(args.pretrained_bert_in)
    args.bert_config = pretrained_bert.config

    # Create model
    logging.info("Creating '{}' model".format(args.model))
    model = get_model(args.model)(args, BERT=pretrained_bert)

    if args.n_gpu > 1 and not args.list_dataparallel:
        logging.info("Parallelising model over {} GPUs".format(args.n_gpu))
        model = DataParallel(model)
    elif args.n_gpu > 1 and args.list_dataparallel:
        logging.info("Parallelising model for list over {} GPUs".format(args.n_gpu))
        model = ListDataParallel(model)

    model.to(args.device)

    logging.info("Reading dataset")
    dataset = get_datareader("{}_{}".format(args.model, args.downstream))(args)
    dataset.load_test_train()

    logging.info("Training model")
    batch_size = args.single_batch * args.n_gpu

    if args.profile_dir is not None:
        if not isinstance(model, DataParallel):
            model.profiled_fit(
                args.profile_dir,
                dataset.fitting(batch_size=batch_size),
                dataset.predicting(batch_size=batch_size),
                args,
                hook_step=hook_step,
            )
        else:
            profiled_parallel_fit(
                args.profile_dir,
                model,
                dataset.fitting(batch_size=batch_size),
                dataset.predicting(batch_size=batch_size),
                args,
                hook_step=hook_step,
            )
    else:
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
