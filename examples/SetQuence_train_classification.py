from torch.utils.tensorboard import SummaryWriter

from nebis.data import MutationDatasetForClassification
from nebis.models import SetQuence
from nebis.utils.args import argument_parser

from transformers import BertModel


def customize_parser(parser):
    parser.add_argument(
        "--pretrained_bert_in",
        default=None,
        type=str,
        required=True,
        help="Pretrained BERT model",
    )
    return parser


def hook_step(step, values):
    for v in values:
        writer.add_scalar(
            v, values[v], step,
        )


if __name__ == "__main__":
    # Argument parser load and customisation
    parser = customize_parser(argument_parser())
    args = parser.parse_args()

    # TensorBoard writer
    writer = SummaryWriter(args.log_dir)

    # Load BERT config
    pretrained_bert = BertModel.from_pretrained(args.pretrained_bert_in)
    args.bert_config = pretrained_bert.config

    # Create model
    model = SetQuence(args, BERT=pretrained_bert)
    model.to(args.device)

    mutation_data = MutationDatasetForClassification(args)
    mutation_data.load()
    """model.fit(
        mutation_data.fitting(), mutation_data.predicting(), args, hook_step=hook_step
    )"""
    model.profiled_fit(
        args.profile_dir,
        mutation_data.fitting(weighted=args.weighted),
        mutation_data.predicting(),
        args,
        hook_step=hook_step,
    )
