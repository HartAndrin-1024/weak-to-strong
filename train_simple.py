import json
import os
import random
import subprocess
from typing import Dict, List, Optional, Callable

import fire
import numpy as np
import torch
from datasets import load_dataset, load_from_disk
from datasets import Dataset, concatenate_datasets

import weak_to_strong.logger as logger
from weak_to_strong.common import get_tokenizer
from weak_to_strong.datasets import (VALID_DATASETS, load_dataset,
                                     tokenize_dataset)
from weak_to_strong.loss import logconf_loss_fn, product_loss_fn, xent_loss
from weak_to_strong.train import ModelConfig, train_model2

# NOTE learning rates are not particularly tuned, work somewhat reasonably at train batch size 32
MODEL_CONFIGS = [
    ModelConfig(
        name="gpt2",
        default_lr=5e-5,
        eval_batch_size=32,
    ),
    ModelConfig(
        name="gpt2-medium",
        default_lr=5e-5,
        eval_batch_size=32,
    ),
    ModelConfig(
        name="gpt2-large",
        default_lr=1e-5,
        eval_batch_size=32,
    ),
    ModelConfig(
        name="gpt2-xl",
        default_lr=1e-5,
        eval_batch_size=2,
        gradient_checkpointing=True,
        # Should use model_parallel on V100s (note: ironically if you have a single V100 it should run,
        # but if you have multiple it won't run without model_parallel because of the overhead of data
        # parallel training).
        model_parallel=(
            torch.cuda.get_device_properties(0).total_memory < 35e9
            and torch.cuda.device_count() > 1
        ),
    ),

    # I should re-run experiments with gpt2-xl at the end, iteration speed is probably too slow with it for all experiments
]
MODELS_DICT: Dict[str, ModelConfig] = {
    model_config.name: model_config for model_config in MODEL_CONFIGS
}


loss_dict = {
    "logconf": logconf_loss_fn(),
    "product": product_loss_fn(),
    "xent": xent_loss(),
}

VALID_LOSSES: List[str] = list(loss_dict.keys())


def get_config_foldername(config: dict) -> str:
    def shorten_key(key: str) -> str:
        return "".join(word[0] for word in key.split("_"))

    def shorten_value(value) -> str:
        if isinstance(value, bool):
            return "1" if value else "0"
        elif isinstance(value, str):
            value = value.split("/")[-1]
            if "_" in value:
                return "_".join(word[:4] for word in value.split("_"))
            else:
                return value
        else:
            return str(value)

    return "-".join(f"{shorten_key(k)}={shorten_value(v)}" for k, v in sorted(config.items()))


def main(
    batch_size: int = 32,
    max_ctx: int = 1024,
    ds_name: str = "boolq",
    loss: str = "xent",
    n_docs: int = 20000,
    n_test_docs: int = 10000,
    model_size: str = "gpt2",
    lr: Optional[float] = None,
    optim: Optional[str] = None,
    epochs: int = 2,
    force_retrain: bool = False,
    seed: int = 0,
    minibatch_size_per_device: Optional[float] = None,
    train_with_dropout: bool = False,
    results_folder: str = "/tmp/results",
    linear_probe: bool = False,
    lr_schedule: str = "cosine_anneal",
    # Note: you can pass either weak_model_size or weak_labels_path. If you pass
    # weak_model_size, we will guess the path to the weak labels based on the weak
    # model. If you pass weak_labels_path, we will use that path instead.
    # If you pass neither, we will train on ground truth.
    weak_model_size: Optional[str] = None,
    weak_labels_path: Optional[str] = None,
    sweep_subfolder: str = "default",
    # Set to a very large value so that by default we don't do any intermediate evals but
    # still do final evals (which requires eval_every to be set to a non-zero, non-None value)
    eval_every: int = 1000000,
    sync_command: Optional[str] = None,
    loss_fn2: Callable = logconf_loss_fn,
    prop_of_gt: str='0',
    hard_questions_only: str='',
    upweight_gt_factor: str='1',
    curriculum: str='',
    med_questions_only: str='',
    weighted_auto: str='',
    online_correction: str='',
):
    curriculum=bool(curriculum)
    upweight_gt_factor=int(upweight_gt_factor) #TODO: there is more work to be done here including sweeps etc
    prop_of_gt=float(prop_of_gt)
    hard_questions_only=bool(hard_questions_only)
    med_questions_only=bool(med_questions_only)
    weighted_auto=bool(weighted_auto)
    online_correction=bool(online_correction)

    # this is per device!
    if minibatch_size_per_device is None:
        minibatch_size_per_device = 1
    assert ds_name in VALID_DATASETS, f"Unknown dataset {ds_name} not in {VALID_DATASETS}"
    assert (
        weak_model_size is None or weak_labels_path is None
    ), "Can't pass both weak_model_size and weak_labels_path"
    model_config = MODELS_DICT[model_size]

    use_default_lr = False
    if lr is None:
        assert (
            batch_size == 32
        ), "Learning rates were tuned on batch size 32, you probably want to sweep LR if you are tuning batch size"
        lr = model_config.default_lr
        use_default_lr = True

    if optim is None:
        optim = model_config.default_optimizer

    # The commented out terms are the ones that should not change final results
    config = {
        "batch_size": batch_size,
        "max_ctx": max_ctx,
        "ds_name": ds_name,
        "loss": loss,
        "n_docs": n_docs,
        "n_test_docs": n_test_docs,
        "model_size": model_size,
        "lr": lr,
        "optim": optim,
        "epochs": epochs,
        # "force_retrain": force_retrain,
        "seed": seed,
        # "minibatch_size_per_device": minibatch_size_per_device,
        "train_with_dropout": train_with_dropout,
        # "results_folder": results_folder,
        "linear_probe": linear_probe,
        "lr_schedule": lr_schedule,
        "eval_every": eval_every,
        # "sweep_subfolder": sweep_subfolder,
    }

    if weak_model_size is not None:
        weak_model_config = config.copy()
        weak_model_config["model_size"] = weak_model_size
        weak_model_config["loss"] = "xent"
        if use_default_lr:
            weak_model_config["lr"] = MODELS_DICT[weak_model_size].default_lr

        weak_model_config_name = get_config_foldername(weak_model_config)

        weak_labels_path = (
            results_folder + "/" + sweep_subfolder + "/" + weak_model_config_name + "/weak_labels"
        )

    eval_batch_size = model_config.eval_batch_size
    random.seed(seed)

    # Load dataset
    dataset = load_dataset(ds_name, seed=seed, split_sizes=dict(train=n_docs, test=n_test_docs))

    # Split the training dataset in half
    train_dataset, test_ds = dataset["train"], dataset["test"]
    # debug=True

    # import ipdb
    # ipdb.set_trace()

    # if debug:
    #     train_dataset=train_dataset[:100]
    #     test_ds=test_ds[:100]

    # ipdb.set_trace()

    if weak_labels_path is None:
        split_data = train_dataset.train_test_split(test_size=0.5, seed=seed)
        train1_ds, train2_ds = split_data["train"], split_data["test"]
        print("len(train1):", len(train1_ds), "len(train2):", len(train2_ds))
        config_name = get_config_foldername(config)
    else:
        if not weak_labels_path.endswith("weak_labels"):
            weak_labels_path = weak_labels_path + "/weak_labels"
        if sync_command is not None:
            sync_command_list = sync_command.split(" ")
            sync_command_list.extend(
                ["download", weak_labels_path.replace("/weak_labels", ""), results_folder]
            )
            print(f"Running sync command: {' '.join(sync_command_list)}")
            result = subprocess.run(sync_command_list, check=True)
            if result.returncode != 0:
                raise RuntimeError(f"Sync command failed with return code {result.returncode}")
        train1_ds = load_from_disk(weak_labels_path)
        train2_ds = None

        weak_model_config = json.load(open(weak_labels_path.replace("weak_labels", "config.json")))
        config["weak_model_size"] = weak_model_config["model_size"]
        config_name = get_config_foldername(config)
        config["weak_model"] = weak_model_config

    save_path = os.path.join(results_folder, sweep_subfolder, config_name)
    save_path=save_path+f'-propgt{prop_of_gt}'+f'-hardquest{hard_questions_only}'+f'-upweightfactor{upweight_gt_factor}'+f'-medquest{med_questions_only}'+f'-curr{curriculum}'
    logger.configure(
        name="{sweep_subfolder}_{config_name}_{datetime_now}",
        save_path=save_path,
        sweep_subfolder=sweep_subfolder,
        config_name=config_name,
    )
    # Tokenize datasets
    tokenizer = get_tokenizer(model_config.name)
    train1_ds = tokenize_dataset(train1_ds, tokenizer, max_ctx)
    test_ds = tokenize_dataset(test_ds, tokenizer, max_ctx)

    def apply_gt_label(data):
        if data['gt_label']==1:
            data['soft_label']=[float(0),float(1)]
        else:
            data['soft_label']=[float(1),float(0)]
        # data['hard_label']=data['soft_label']
        return data
    if train2_ds:
        train2_ds = tokenize_dataset(train2_ds, tokenizer, max_ctx)
    elif curriculum:
        train1_ds=train1_ds.shuffle(seed=seed)
        split_datasets=train1_ds.train_test_split(test_size=prop_of_gt, seed=seed)
        to_adj=split_datasets['test']
        to_stay=split_datasets['train']
        to_adj=to_adj.map(apply_gt_label)
        
        to_adj_split1=to_adj.train_test_split(test_size=.4, seed=seed)
        to_adj_part1=to_adj_split1['test']
        to_adj_split2=to_adj_split1['train'].train_test_split(test_size=.5,seed=seed)
        to_adj_part2=to_adj_split2['test']
        to_adj_split3=to_adj_split2['train'].train_test_split(test_size=.6666,seed=seed)
        to_adj_part3=to_adj_split3['test']
        to_adj_part4=to_adj_split3['train']

        to_stay_split1=to_stay.train_test_split(test_size=.25, seed=seed)
        to_stay_part1=to_stay_split1['test']
        to_stay_split2=to_stay_split1['train'].train_test_split(test_size=.33333,seed=seed)
        to_stay_part2=to_stay_split2['test']
        to_stay_split3=to_stay_split2['train'].train_test_split(test_size=.5,seed=seed)
        to_stay_part3=to_stay_split3['test']
        to_stay_part4=to_stay_split3['train']

        final_part1=concatenate_datasets([to_adj_part1,to_stay_part1])
        final_part1=final_part1.shuffle(seed=seed)
        final_part2=concatenate_datasets([to_adj_part2,to_stay_part2])
        final_part2=final_part2.shuffle(seed=seed)
        final_part3=concatenate_datasets([to_adj_part3,to_stay_part3])
        final_part3=final_part3.shuffle(seed=seed)
        final_part4=concatenate_datasets([to_adj_part4,to_stay_part4])
        final_part4=final_part4.shuffle(seed=seed)

        train1_ds=concatenate_datasets([final_part4,final_part3,final_part2,final_part1])

    elif upweight_gt_factor!=1:
        train1_ds=train1_ds.shuffle(seed=seed)
        split_datasets=train1_ds.train_test_split(test_size=prop_of_gt, seed=seed)
        to_adj=split_datasets['test']
        to_stay=split_datasets['train']
        to_adj=to_adj.map(apply_gt_label)
        to_add_adj=concatenate_datasets([to_adj]*upweight_gt_factor)
        train1_ds=concatenate_datasets([to_add_adj,to_stay])
        train1_ds=train1_ds.shuffle(seed=seed)
    elif hard_questions_only:
        def amount_wrong(data):
            data['amount_wrong']=(1-data['acc'])*(max(data['soft_label']))
            return data
        train1_ds=train1_ds.map(amount_wrong)
        train1_ds=train1_ds.sort('amount_wrong', reverse=True)
        first_n=train1_ds.select(range(int(len(train1_ds)*prop_of_gt)))
        remaining=train1_ds.select(range(int(len(train1_ds)*prop_of_gt),len(train1_ds)))
        first_n=first_n.map(apply_gt_label)
        train1_ds=concatenate_datasets([first_n,remaining])
        train1_ds=train1_ds.shuffle(seed=seed)
    elif med_questions_only:
        def amount_uncertain(data):
            data['amount_uncertain']=(abs(data['soft_label'][0]-0.5))
            return data
        train1_ds=train1_ds.map(amount_uncertain)
        train1_ds=train1_ds.sort('amount_uncertain')
        first_n=train1_ds.select(range(int(len(train1_ds)*prop_of_gt)))
        remaining=train1_ds.select(range(int(len(train1_ds)*prop_of_gt),len(train1_ds)))
        first_n=first_n.map(apply_gt_label)
        train1_ds=concatenate_datasets([first_n,remaining])
        train1_ds=train1_ds.shuffle(seed=seed)
    elif not online_correction:
        if prop_of_gt!=0:
            train1_ds=train1_ds.shuffle(seed=seed)
            split_datasets=train1_ds.train_test_split(test_size=prop_of_gt, seed=seed)
            to_adj=split_datasets['test']
            to_stay=split_datasets['train']
            to_adj=to_adj.map(apply_gt_label)
            train1_ds=concatenate_datasets([to_stay,to_adj])
        train1_ds=train1_ds.shuffle(seed=seed) # Note to self: have to rerun these experiments, the first one was bugged and wasn't shuffling properly
    loss_fn = loss_dict[loss]
    print(f"Training model model, size {model_size}")
    test_results, weak_ds = train_model2(
        model_config,
        train1_ds,
        test_ds,
        inference_ds=train2_ds,
        batch_size=batch_size,
        save_path=save_path,
        loss_fn=loss_fn,
        lr=lr,
        epochs=epochs,
        force_retrain=force_retrain,
        eval_batch_size=eval_batch_size,
        minibatch_size_per_device=minibatch_size_per_device,
        train_with_dropout=train_with_dropout,
        linear_probe=linear_probe,
        lr_schedule=lr_schedule,
        optimizer_name=optim,
        eval_every=eval_every,
        weighted_auto=weighted_auto,
        online_correction=online_correction,
        prop_of_gt=prop_of_gt,
    )

    if weak_ds is not None:
        weak_ds.save_to_disk(save_path + "/" + "weak_labels")

    acc = np.mean([x["acc"] for x in test_results])
    res_dict = {"accuracy": acc}
    print("accuracy:", acc)

    # # logconf_loss=logconf_loss_fn()
    # def logconf_loss(logits,labels,step_frac):
    #     logits = logits.float()
    #     labels = labels.float()
    #     coef = 1.0
    #     coef = coef * self.aux_coef
    #     preds = torch.softmax(logits, dim=-1)
    #     mean_weak = torch.mean(labels, dim=0)
    #     assert mean_weak.shape == (2,)
    #     threshold = torch.quantile(preds[:, 0], mean_weak[1])
    #     strong_preds = torch.cat(
    #         [(preds[:, 0] >= threshold)[:, None], (preds[:, 0] < threshold)[:, None]],
    #         dim=1,
    #     )
    #     target = labels * (1 - coef) + strong_preds.detach() * coef
    #     loss = torch.nn.functional.cross_entropy(logits, target, reduction="none")
    #     return loss.mean()
    # logits=[x['logits'] for x in test_results]
    # labels=[x['soft_label'] for x in test_results]

    # logconf=logconf_loss(torch.tensor(logits),torch.tensor(labels),1)
    # config['logconf']=logconf

    # import ipdb
    # ipdb.set_trace()

    with open(os.path.join(save_path, f"config.json"), "w") as f:
        json.dump(config, f, indent=2)

    with open(os.path.join(save_path, f"results_summary.json"), "w") as f:
        json.dump(res_dict, f, indent=2)

    if sync_command is not None:
        print("Syncing results to remote storage...")
        try:
            sync_command_list = sync_command.split(" ")
            sync_command_list.extend(["upload", save_path, results_folder])
            print(f"Running sync command: {' '.join(sync_command_list)}")
            result = subprocess.run(sync_command_list, check=True)
            if result.returncode != 0:
                raise RuntimeError(f"Sync command failed with return code {result.returncode}")
        except Exception as e:
            raise RuntimeError("Failed to sync results to remote storage.") from e


if __name__ == "__main__":
    fire.Fire(main)
