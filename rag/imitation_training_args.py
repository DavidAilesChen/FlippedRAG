import os
import sys
curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(curdir))
from dataclasses import dataclass, field
from typing import Optional

current_file = __file__
parent_dir = os.path.dirname(current_file)
grandparent_dir = os.path.dirname(parent_dir)
model_dir = os.path.dirname(os.path.dirname(grandparent_dir))+"/model_hub"
BERT_LM_MODEL_DIR = '/wiki103/bert/'

@dataclass
class TrainingArguments:
    model_name_or_path: Optional[str] = field(
        default="MiniLM",
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    mode: Optional[str] = field(
        default="train",
        metadata={
            "help": (
                "Processing phase."
            )
        },
    )
    method: Optional[str] = field(
        default="aggressive",
        metadata={
            "help": (
                "Method mentioned."
            )
        },
    )
    data_name:Optional[str] = field(
        default="dl",
        metadata={
            "help": (
                "Data used when filtering successful trigger."
            )
        },
    )
    target: Optional[str] = field(
        default="mini",
        metadata={
            "help": (
                "base model. mini or nb_bert."
            )
        },
    )
    target_type:Optional[str] = field(
        default="none",
        metadata={
            "help": (
                "base model."
            )
        },
    )
    epoch_num: Optional[int] = field(
        default=6,
        metadata={
            "help": (
                "The amount of collision generating epochs."
            )
        },
    )
    nature:  Optional[bool] = field(
        default=True,
        metadata={
            "help": (
                "Define whether to generate natural collision."
            )
        },
    )
    pat: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Define whether to generate pat trigger."
            )
        },
    )
    regularize:  Optional[bool] = field(
        default=True,
        metadata={
            "help": (
                "Use regularize to decrease perplexity."
            )
        },
    )
    nsp:  Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Use Next sentence prediction to enhance similarity."
            )
        },
    )
    num_beams: Optional[int] = field(
        default=10,
        metadata={
            "help": (
                "The number of beams."
            )
        },
    )
    num_filters: Optional[int] = field(
        default=100,
        metadata={
            "help": (
                "The number of filters."
            )
        },
    )
    patience_limit: Optional[int] = field(
        default=2,
        metadata={
            "help": (
                "Patience for early stopping."
            )
        },
    )
    perturb_iter: Optional[int] = field(
        default=5,
        metadata={
            "help": (
                "PPLM iteration"
            )
        },
    )
    lm_model_dir: Optional[str] = field(
        default=BERT_LM_MODEL_DIR,
        metadata={
            "help": (
                "bert path."
            )
        },
    )
    topk: Optional[int] = field(
        default=100,
        metadata={
            "help": (
                "top N selection."
            )
        },
    )
    seq_len: Optional[int] = field(
        default=6,
        metadata={
            "help": (
                "length of sequence."
            )
        },
    )
    lr: Optional[float] = field(
        default=0.001,
        metadata={
            "help": (
                "learning rate."
            )
        },
    )
    kl_scale: Optional[float] = field(
        default=0.00,
        metadata={
            "help": (
                "KL divergence coefficient."
            )
        },
    )
    stemp: Optional[float] = field(
        default=1.0,
        metadata={
            "help": (
                "temperature of softmax."
            )
        },
    )
    verbose:  Optional[bool] = field(
        default=True,
        metadata={
            "help": (
                "Print every iteration."
            )
        },
    )
    beta: Optional[float] = field(
        default=0.0,
        metadata={
            "help": (
                "Coefficient for language model loss."
            )
        },
    )
    min_len: Optional[int] = field(
        default=5,
        metadata={
            "help": (
                "Min sequence length."
            )
        },
    )
    max_iter: Optional[int] = field(
        default=5,
        metadata={
            "help": (
                "Max number of iteration."
            )
        },
    )
    type: Optional[str] = field(
        default="triple",
        metadata={
            "help": (
                "msmarce data type."
            )
        },
    )
    proportion: Optional[float] = field(
        default=1,
        metadata={
            "help": (
                "proportion of train data."
            )
        },
    )
    adv_proportion: Optional[float] = field(
        default=0.5,
        metadata={
            "help": (
                "proportion of adversarial samples."
            )
        },
    )
    seed: Optional[str] = field(
        default=45,
        metadata={
            "help": (
                "random seed."
            )
        },
    )
    num_epochs: Optional[int] = field(
        default=10,
        metadata={
            "help": (
                "Number of epochs for training."
            )
        },
    )
    num_training_instances:Optional[int] = field(
        default=-1,
        metadata={
            "help": (
                "Number of training instances for training (if num_training_instances != -1 then num_epochs is ignored)."
            )
        },
    )
    validate_every_epochs:Optional[int] = field(
        default=1,
        metadata={
            "help": (
                "Run validation every <validate_every_epochs> epochs."
            )
        },
    )
    validate_every_steps:Optional[int] = field(
        default=100,
        metadata={
            "help": (
                "Run validation every <validate_every_steps> steps."
            )
        },
    )
    num_validation_batches:Optional[int] = field(
        default=64,
        metadata={
            "help": (
                "Run validation for a sample of <num_validation_batches>. To run on all instances use -1."
            )
        },
    )
    train_batch_size:Optional[int] = field(
        default=24,
        metadata={
            "help": (
                "Training batch size."
            )
        },
    )
    val_batch_size:Optional[int] = field(
        default=32,
        metadata={
            "help": (
                "Validation and test batch size."
            )
        },
    )
    sample_data:Optional[int] = field(
        default=-1,
        metadata={
            "help": (
                "Amount of data to sample for training and eval. If no sampling required use -1."
            )
        },
    )
    use_dev_triple:Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "whether use dev triples to select the best model for pseudo label and extract."
            )
        },
    )
    pseudo_final:Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "whether use the best saved model to pseudo datasets"
            )
        },
    )
    max_seq_len:Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "Maximum sequence length for the inputs."
            )
        },
    )
    max_grad_norm:Optional[int] = field(
        default=1,
        metadata={
            "help": (
                "Max gradient normalization."
            )
        },
    )
    lr_train:Optional[float] = field(
        default=3e-6,
        metadata={
            "help": (
                "Learning rate in training."
            )
        },
    )
    num_sims:Optional[int] = field(
        default=300,
        metadata={
            "help": (
                "number of PAT augmentation words."
            )
        },
    )
    accumulation_steps:Optional[float] = field(
        default=1,
        metadata={
            "help": (
                "gradient accumulation."
            )
        },
    )
    warmup_portion:Optional[float] = field(
        default=0.1,
        metadata={
            "help": (
                "warmup portion."
            )
        },
    )
    lambda_1:Optional[float] = field(
        default=0.1,
        metadata={
            "help": (
                "Coefficient for language model loss."
            )
        },
    )
    lambda_2:Optional[float] = field(
        default=0.8,
        metadata={
            "help": (
                "Coefficient for language model loss."
            )
        },
    )
    loss_function:Optional[str] = field(
        default="label-smoothing-cross-entropy",
        metadata={
            "help": (
                "Loss function (default is 'cross-entropy')."
            )
        },
    )
    smoothing:Optional[float] = field(
        default=0.1,
        metadata={
            "help": (
                "Smoothing hyperparameter used only if loss_function is label-smoothing-cross-entropy."
            )
        },
    )
    transformer_model:Optional[str] = field(
        default="cross-encoder/ms-marco-MiniLM-L-12-v2",
        metadata={
            "help": (
                "Model specific name."
            )
        },
    )
    model_name:Optional[str] = field(
        default='pairwise-minilm-ranker',
        metadata={
            "help": (
                "Training model name."
            )
        },
    )
