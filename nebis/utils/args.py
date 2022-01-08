import argparse
from nebis.models.downstream import list_downstream
from nebis.models.pooling import list_pooler
from nebis.models import list_models


def argument_parser():
    parser = argparse.ArgumentParser()

    _models = list_models()
    parser.add_argument(
        "--model",
        default="setquence" if "setquence" in _models else _models[0],
        choices=_models,
        type=str,
        help="What model to train; e.g., setquence",
    )
    _downstream_tasks = list_downstream()
    parser.add_argument(
        "--downstream",
        default="classification"
        if "classification" in _downstream_tasks
        else _downstream_tasks[0],
        choices=_downstream_tasks,
        type=str,
        help="Type of downstream task",
    )
    parser.add_argument(
        "--kmer",
        default=6,
        type=int,
        help="For DNABERT: k-mer size of the sequence dataset (3, 4, 5 or 6)",
    )
    parser.add_argument(
        "--sequence_length",
        default=64,
        type=int,
        help="For DNABERT: Maximum sequence length (# tokens)",
    )
    parser.add_argument(
        "--max_numeric",
        default=60483,
        type=int,
        help="Number of measured expression level datapoints",
    )
    parser.add_argument(
        "--weighted",
        dest="weighted",
        action="store_true",
        help="The datasampler will be reweighted according to the number of cases per category",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        type=str,
        help="The device to perform training on. Choose between 'cuda' or 'cpu'",
    )
    _poolers = list_pooler()
    parser.add_argument(
        "--pooling_sequence",
        default="PMA" if "PMA" in _poolers else _poolers[0],
        choices=_poolers,
        type=str,
        help="Pooling method for sequence data",
    )
    parser.add_argument(
        "--pooling_numeric",
        default="PMA" if "PMA" in _poolers else _poolers[0],
        choices=_poolers,
        type=str,
        help="Pooling method for discretised numeric data",
    )
    parser.add_argument(
        "--activation",
        default="sigmoid",
        choices=["sigmoid", "relu"],
        type=str,
        help="Activation function, can be Sigmoid, ReLU",
    )
    parser.add_argument(
        "--checkpoint_interval",
        default=5,
        type=int,
        required=False,
        help="The checkpointing frequency respect to the number of training epochs",
    )
    parser.add_argument(
        "--pretrained_bert_in",
        default=None,
        type=str,
        required=True,
        help="Pretrained BERT model",
    )
    parser.add_argument(
        "--model_out",
        default=None,
        type=str,
        required=True,
        help="Where to save the model, checkpoints and intermediate calculations (numpy arrays)",
    )
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The location of the train.torch and dev.torch files with the dataset",
    )
    parser.add_argument(
        "--max_mutations",
        default=500,
        type=int,
        help="Maximum number of mutations in the dataset (for splitting into n BERT networks)",
    )
    parser.add_argument(
        "--single_batch",
        default=1,
        type=int,
        help="Batch size for a single compute unit (e.g., 1 mutome per GPU is often around 10GB)",
    )
    parser.add_argument(
        "--log_dir",
        default=None,
        type=str,
        required=True,
        help="Directory for logging",
    )
    parser.add_argument(
        "--profile_dir",
        default=None,
        type=str,
        required=True,
        help="The location of the profiling",
    )
    parser.add_argument(
        "--num_classes",
        default=2,
        type=int,
        help="Number of output classes for the last classification module",
    )
    parser.add_argument(
        "--num_times",
        default=10,
        type=int,
        help="Number of output time intervals for last survival module",
    )
    parser.add_argument(
        "--learning_rate",
        default=0.00001,
        type=float,
        help="Learning rate of the Adam optimizer",
    )
    parser.add_argument(
        "--weight_decay",
        default=0,
        type=float,
        help="Weight decay for the ADAM optimizer (L2 regularisation). Does not have effect with Batch Norm.",
    )
    parser.add_argument(
        "--epochs", default=40, type=int, help="Number of training epochs",
    )
    parser.add_argument(
        "--k_seeds",
        default=1,
        type=int,
        help="Number of seeds for the Multi-head attention layer(s)",
    )
    parser.add_argument(
        "--p_dropout",
        default=0.3,
        type=float,
        help="Probability of dropout (while training) of the classification path",
    )
    parser.add_argument(
        "--mutome_heads",
        default=12,
        type=int,
        help="Number of attention heads for PMA, MAB or ISAB",
    )
    parser.add_argument(
        "--embedding_size",
        default=768,
        type=int,
        help="Dimensionality of the embeddings",
    )
    parser.add_argument(
        "--hidden_size",
        default=768,
        type=int,
        help="Dimensionality of the hidden layer",
    )
    parser.add_argument(
        "--finetune",
        dest="finetune",
        action="store_true",
        help="Enable BERT fine-tuning",
    )
    parser.add_argument(
        "--finetune_max_mutations",
        default=110,
        type=int,
        help="Maximum mutations for the fine-tuning. Can be equal or less than --max_mutations",
    )
    parser.add_argument(
        "--warmup_percent",
        default=0.1,
        type=float,
        help="Fraction of steps for linear warmup schedule",
    )
    parser.add_argument(
        "--digitize_bins",
        default=50,
        type=int,
        help="Vocabulary size (for omics embedding)",
    )
    parser.add_argument(
        "--digitize_min",
        default=-0.0001,
        type=int,
        help="Minimum expression level to map (discretisation)",
    )
    parser.add_argument(
        "--digitize_max",
        default=1,
        type=int,
        help="Maximum expression level to map (discretisation)",
    )
    parser.add_argument(
        "--n_gpu", default=1, type=int, help="Number of GPUs for training",
    )
    parser.add_argument(
        "--seed", default=2022, type=int, help="Random seed",
    )

    return parser
