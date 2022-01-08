# nebis

*by Daniel León-Periñán @ ZIH, TU Dresden*

****nebis**** is a collection of Neural Networks for learning Biological data as Set Representations. In this repository, we release the source code implementing SetQuence and SetOmic neural networks, for training and testing of classification and survival as downstream tasks. Moreover, we provide tools for Explainable AI (XAI) analysis of primary attribution.

## First Steps

### A. Using _nebis_ from a container

We recommend using the provided Docker container, that is already set-up with all dependencies. This container is based on with CUDA 10.0 running under Ubuntu 18.04.

```
docker pull ghcr.io/danilexn/nebis/nebis:master
docker run danilexn/nebis:latest
```

### B. Installing _nebis_

_nebis_ can be installed on Unix-based or Windows systems supporting Python >=3.6. Please, make sure this requirement is met. We recommend installing _nebis_ in a python virtual environment with [Anaconda](https://docs.anaconda.com/anaconda/install/linux/).

#### 1.1 Create and activate a new virtual environment

```
conda create -n nebis python=3.6
conda activate nebis
```

#### 1.2 Install the package and other requirements

(Required) First, install DNABERT and its dependencies. DNABERT is a core component of _SetQuence_ and _SetOmic_, as it is the encoder for biological sequence data. Installing all requirements for this package will solve most dependencies for _nebis_.

```
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch

git clone https://github.com/jerryji1993/DNABERT
cd DNABERT
python3 -m pip install --editable .
cd examples
python3 -m pip install -r requirements.txt
```

(Required) Analogously, proceed to install _nebis_

```
git clone https://github.com/danilexn/nebis
cd nebis
python3 -m pip install .
```

## Training _SetQuence_ and _SetOmic_

#### 2.1 Data processing

Two different types of model have been trained for _SetQuence_ and _SetOmic_ architectures, from the pan-cancer TCGA dataset: (i) for cancer type prediction, and (ii) for per-patient survival profile prediction.

We provide our full dataset in this link, as a `<code>`.torch`</code>` file containing a tuple of `<code>`torch.tensor`</code>` data structures, which can be read as:

```
import torch

datatuple = torch.load("dataset.torch")

# for n patients
datatuple[0] # sequence data: an integer (n*m*s) torch.tensor, where each patient n has a maximum of with m sequences of length s
datatuple[1] # attention masks: an integer (n*m*s) torch.tensor, for 0/1 masking of the sequence data
datatuple[2] # token type id: an integer (n*m*s) torch.tensor
datatuple[3] # expression: a float (n*m) torch.tensor containing standardised expression quantification for m expression *loci*
datatuple[4] # label: an integer (n) torch.tensor containing the cancer type labels, 0-32; 0-31 are cancer types and 32 are healthy control samples.
datatuple[5] # barcode: a string (n) torch.tensor containing the sample barcodes for the specific sample n
datatuple[6] # event: a binary (n) torch.tensor containing the indicator of a measured event (i.e., alive or dead)
datatuple[7] # time: an integer (n) torch.tensor containing the measurement time of an event
```

In summary, the dataset consists of per-patient and standardised mRNA-sequencing *transcriptome* profiles, and per-patient *mutomes* represented by sequence contexts around mutation events. Sequences, with a maximum length of $s=64bp$, were transformed into k-mer representations, with $k=6$. Then, individual k-mers were converted into integers by tokenization, with the tools provided in DNABERT.

_nebis_ provides, in `<code>`nebis.data.dataset`</code>`, several structures to automatically parse and adapt this data structure into a format that can be used for training and inference.

#### 2.2 Training _SetQuence_

Starting from our pre-built dataset, _SetQuence_ can be trained by combined fine-tuning of a DNABERT model, and subsequent training of the _pooling_ units.

```
cd examples

export KMER=6
export SEQ_LEN=64
# specify the correct directories in your machine
export DATA_DIR=~/setquence_data # i.e., downloaded from link above; the directory must contain both train.torch and dev.torch files, for training and testing
export PRETRAINED_BERT=~/dnabert_6_64
export OUTPUT_DIR=~/setquence_model_output
export LOGGING_DIR=~/logs

python nebis_train.py \
    --model "setquence" \
    --downstream "classification \
    --output_dir $OUTPUT_PATH \
    --model_out models \
    --data_dir data \
    --profile_dir $LOGGING_PATH \
    --log_dir $LOGGING_PATH \
    --pretrained_bert_in $PRETRAINED_BERT \
    --num_classes 33
```

(optional) Performance profiles can be obtained (also for _SetOmic_) at a directory <code>PROFILE_DIR</code> with the <code>--profile_dir $PROFILE_DIR</code> flag. This will store <code>json</code> files that can be explored with TensorBoard Profiler.
#### 2.2 Training _SetOmic_

Training _SetOmic_ is equivalent to training _SetQuence_; please, make sure that the dataset must contain multiple input data features (i.e., expression data and sequence data).

```
cd examples

export KMER=6
export SEQ_LEN=64
# specify the correct directories in your machine
export DATA_DIR=~/setquence_data
export PRETRAINED_BERT=~/dnabert_6_64 # or a SetQuence model
export OUTPUT_DIR=~/setomic_model_output
export LOGGING_DIR=~/logs

python nebis_train.py \
    --model "setomic" \
    --downstream "classification \
    --output_dir $OUTPUT_PATH \
    --model_out models \
    --data_dir data \
    --profile_dir $LOGGING_PATH \
    --log_dir $LOGGING_PATH \
    --pretrained_bert_in $PRETRAINED_BERT \
    --num_classes 33
```

## Inference with _SetQuence_ and _SetOmic_

After the model is trained, predictions can be obtained for further data with the following command

```$
# specify the correct directories in your machine
export MODEL_DIR=~/setomic_model_output # or a trained SetQuence model
export DATA_PATH=~/setquence_data/data.torch # a specific .torch file
export PREDICTION_DIR=~/predictions

python nebis_predict.py \
    --model $MODEL_DIR \
    --data_dir $DATA_DIR \
    --prediction_dir $PREDICTION_DIR
```

With the above command, either _SetQuence_ or _SetOmic_ model will be loaded from `MODEL_DIR` , and predictions made on the dataset at `DATA_PATH` will be saved at the directory `PREDICTION_DIR`.

## XAI analysis

Work in progress


