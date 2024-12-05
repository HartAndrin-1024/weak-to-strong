import os
import subprocess
import sys
from typing import List, Union

import fire


def main(model_sizes: Union[List[str], str], prop_of_gt: str='0', hard_questions_only='', upweight_gt_factor='1' ,med_questions_only='',weighted_auto='1.0',online_correction='',curriculum='',**kwargs):
    if isinstance(model_sizes, str):
        model_sizes = model_sizes.split(",")
    assert (
        "weak_model_size" not in kwargs
        and "model_size" not in kwargs
        and "weak_labels_path" not in kwargs
    ), "Need to use model_sizes when using sweep.py"
    curriculum=str(curriculum)
    weighted_auto=str(weighted_auto)
    prop_of_gt=str(prop_of_gt)
    hard_questions_only=str(hard_questions_only)
    upweight_gt_factor=str(upweight_gt_factor)
    med_questions_only=str(med_questions_only)
    online_correction=str(online_correction)
    basic_args = [sys.executable, os.path.join(os.path.dirname(__file__), "train_simple.py")]
    for key, value in kwargs.items():
        basic_args.extend([f"--{key}", str(value)])

    print("Running transfer models")
    for pack in [('True','1','0.075'),('True','1','0.225'),('','4','0.075'),('','4','0.225')]:
        for i in range(len(model_sizes)):
            for j in range(i, len(model_sizes)):
                prop_of_gt=pack[2]
                upweight_gt_factor=pack[1]
                med_questions_only=pack[0]
                weak_model_size = model_sizes[i]
                strong_model_size = model_sizes[j]
                print(f"Running weak {weak_model_size} to strong {strong_model_size}")
                # import ipdb
                # ipdb.set_trace()
                subprocess.run(
                    basic_args
                    + ["--weak_model_size", weak_model_size, "--model_size", strong_model_size, "--prop_of_gt", prop_of_gt, "--hard_questions_only",hard_questions_only, "--upweight_gt_factor", upweight_gt_factor, "--med_questions_only", med_questions_only, "--weighted_auto", weighted_auto,"--online_correction",online_correction, '--curriculum', curriculum],
                    check=True,
                )


if __name__ == "__main__":
    fire.Fire(main)
