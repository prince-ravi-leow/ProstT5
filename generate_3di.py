"""Generate 3di from aa seq or protein struct (ProstT5 + weights and foldseek installation)"""

# Adapted from https://github.com/mheinzinger/ProstT5/tree/main/scripts [predict_3Di_encoderOnly.py , translate.py]

# import re
from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path

from scripts.predict_3Di_encoderOnly import get_embeddings
from scripts.translate import get_torch_device, translate

# import torch
from tqdm import tqdm

# from transformers import AutoModelForSeq2SeqLM, T5Tokenizer, set_seed
from transformers import set_seed
from utils import _run_command, check_input


def from_struct_get_3di(input_dir, output_dir):
    queryDB = output_dir / "queryDB"
    queryDB_h = output_dir / "queryDB_h"
    queryDB_ss_h = output_dir / "queryDB_ss_h"
    queryDB_ss = output_dir / "queryDB_ss"
    queryDB_ss_fasta = output_dir / "queryDB_ss.fasta"
    commands = [
        f"foldseek createdb {input_dir.as_posix()} {queryDB.as_posix()}",
        f"foldseek lndb {queryDB_h.as_posix()} {queryDB_ss_h.as_posix()}",
        f"foldseek convert2fasta {queryDB_ss.as_posix()} {queryDB_ss_fasta.as_posix()}",
    ]
    for cmd in commands:
        _run_command(cmd)


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


def parse_args():
    help_msg = """
    Generate 3di alphabet from either AA seq or prot struct, ONLY submit directory
    
    Requires ProstT5 + weights installation and foldseek 
    
    DOES *NOT* SUPPORT ???

    """
    parser = ArgumentParser(description=help_msg, formatter_class=RawTextHelpFormatter)
    parser.add_argument("input_dir", type=str, help="Input sequences or structures")
    parser.add_argument(
        "--mode",
        choices=("seq", "struct"),
        required=True,
        help="Get 3Di alphabet from either aa SEQ or protein STRUCT",
    )
    parser.add_argument(
        "--encoderOnly",
        action="store_true",
        default=True,
        help="FAST Use 'encoderOnly' mode",
    )
    parser.add_argument(
        "--translate",
        action="store_true",
        default=False,
        help="FAST Use 'encoderOnly' mode",
    )
    parser.add_argument(
        "--device",
        choices=("cuda:0", "cuda:1", "mps", "cpu"),
        default="cuda:0",
        help="Choose which device to run on (useful for distributing jobs accross multiple GPU's)",
    )

    parser.add_argument("--output-dir", default=None, help="Output directory")

    args = parser.parse_args()

    if args.seq:
        assert (
            not args.encoderOnly and args.translate
        ), "Please select EITHER encoderOnly or translate mode"

    return args


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    mode = args.mode
    args_device = args.device

    if not args.output_dir:
        output_dir = Path(input_dir / "output/")
        output_dir.mkdir(exist_ok=True)

    if mode == "struct":
        check_input(input_dir, "pdb")
        from_struct_get_3di(input_dir, output_dir)

    if mode == "seq":
        check_input(input_dir, "fasta")
        device = get_torch_device(args_device)
        seq_files = list(input_dir.glob("*.fasta"))
        for seq_file in tqdm(seq_files):
            outfile_name = f"{Path(seq_file).name}_3di.fasta"
            if args.translate:
                translate(
                    in_path=seq_file,
                    out_path=outfile_name,
                    model_dir="Rostlab/ProstT5",
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
                    model_dir="Rostlab/ProstT5",
                    split_char="!",
                    id_field=0,
                    half_precision=1,
                    output_probs=False,
                )


if __name__ == "__main__":
    main()
