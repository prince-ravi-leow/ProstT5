#!/usr/bin/env python3

# """Generate 3di from aa seq or protein struct (ProstT5 + weights and foldseek installation)"""

# Adapted from https://github.com/mheinzinger/ProstT5/tree/main/scripts [predict_3Di_encoderOnly.py , translate.py]

import shutil
import uuid
from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path

from scripts.predict_3Di_encoderOnly import get_embeddings
from scripts.translate import get_torch_device, translate
from tqdm import tqdm
from transformers import set_seed
from utils import _run_command, check_input

set_seed(42)  # ensure reproducability

# config used for translate.py if input is AA (as opposed to 3di)
aa2ss = {
    "do_sample": True,
    "num_beams": 3,
    "top_p": 0.95,
    "temperature": 1.2,
    "top_k": 6,
    "repetition_penalty": 1.2,
}


def from_struct_get_3di(input_dir, output_dir):
    tmp_dir = Path(output_dir / "tmp_foldseek")
    tmp_dir.mkdir(exist_ok=True)

    queryDB = tmp_dir / "queryDB"
    queryDB_h = tmp_dir / "queryDB_h"
    queryDB_ss_h = tmp_dir / "queryDB_ss_h"
    queryDB_ss = tmp_dir / "queryDB_ss"
    queryDB_ss_fasta = tmp_dir / "queryDB_ss.fasta"
    commands = [
        f"foldseek createdb {input_dir.as_posix()} {queryDB.as_posix()}",
        f"foldseek lndb {queryDB_h.as_posix()} {queryDB_ss_h.as_posix()}",
        f"foldseek convert2fasta {queryDB_ss.as_posix()} {queryDB_ss_fasta.as_posix()}",
    ]
    for cmd in commands:
        _run_command(cmd)

    run_id = str(uuid.uuid4())
    output_file_path = output_dir / f"3di_from_struct_{run_id}.fasta"

    shutil.copy(queryDB_ss_fasta, output_file_path)
    print(f"Results written to {str(output_file_path)}")


def create_arg_parser():
    help_msg = """
    Generate 3di alphabet from either AA seq or prot struct, ONLY submit directory
    
    Requires ProstT5 + weights installation and foldseek 
    """
    parser = ArgumentParser(description=help_msg, formatter_class=RawTextHelpFormatter)
    parser.add_argument("input_dir", type=str, help="Input sequences or structures")
    parser.add_argument("output_dir", type=str, help="Output directory")
    parser.add_argument(
        "--mode",
        choices=("seq", "struct"),
        required=True,
        help="Get 3Di alphabet from either aa SEQ or protein STRUCT",
    )
    parser.add_argument(
        "--encoderOnly",
        action="store_true",
        default=False,
        help="Use 'encoderOnly' mode (FASTER, LESS ACCURATE)",
    )
    parser.add_argument(
        "--translate",
        action="store_true",
        default=False,
        help="Use 'encoderOnly' mode (SLOWER, MORE ACCURATE)",
    )
    parser.add_argument(
        "--device",
        choices=("cuda:0", "cuda:1", "mps", "cpu"),
        default="cuda:0",
        help="Choose which device to run on (useful for distributing jobs accross multiple GPU's)",
    )

    args = parser.parse_args()

    if args.encoderOnly:
        assert not args.translate, "Please select EITHER encoderOnly or translate mode"
    elif args.translate:
        assert not args.encoderOnly, (
            "Please select EITHER encoderOnly or translate mode"
        )

    return args


def main():
    args = create_arg_parser()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    mode = args.mode
    args_device = args.device

    if output_dir.is_dir():
        print("Result directory already exists! - careful for overwriting results!")
    else:
        output_dir.mkdir(exist_ok=True)

    if mode == "struct":
        check_input(input_dir, "pdb")
        from_struct_get_3di(input_dir, output_dir)

    if mode == "seq":
        check_input(input_dir, "fasta")
        device = get_torch_device(args_device)
        seq_files = list(input_dir.glob("*.fasta"))
        print(seq_files)
        for seq_file in tqdm(seq_files):
            outfile_name = f"{Path(seq_file).stem}_3di.fasta"
            if args.translate:
                translate(
                    in_path=seq_file,
                    out_path=outfile_name,
                    model_dir="/mnt/common/prot/models/huggingface/hub/models--Rostlab--ProstT5_fp16/snapshots/07a6547d51de603f1be84fd9f2db4680ee535a86",
                    split_char="!",
                    id_field=0,
                    half_precision=1,
                    is_3Di=False,
                    device=device,
                    gen_kwargs=aa2ss,
                )
            if args.encoderOnly:
                get_embeddings(
                    seq_path=seq_file,
                    out_path=outfile_name,
                    model_dir="/mnt/common/prot/models/huggingface/hub/models--Rostlab--ProstT5_fp16/snapshots/07a6547d51de603f1be84fd9f2db4680ee535a86",
                    device=device,
                    split_char="!",
                    id_field=0,
                    half_precision=1,
                    output_probs=False,
                )


if __name__ == "__main__":
    main()
