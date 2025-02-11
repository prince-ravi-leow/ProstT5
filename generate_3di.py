"""Generate 3di from aa seq or protein struct (ProstT5 + weights and foldseek installation)"""

import uuid
from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path
from tempfile import TemporaryDirectory

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
    """Run foldseek through subprocess on directory containing .pdb files, get 3di"""

    with TemporaryDirectory(dir=output_dir) as tmp_dir:
        tmp_dir = Path(tmp_dir)
        run_id = str(uuid.uuid4())
        output_file_path = output_dir / f"3di_from_struct_{run_id}.fasta"

        queryDB = tmp_dir / "queryDB"
        queryDB_h = tmp_dir / "queryDB_h"
        queryDB_ss_h = tmp_dir / "queryDB_ss_h"
        queryDB_ss = tmp_dir / "queryDB_ss"

        commands = [
            f"foldseek createdb {input_dir} {queryDB}",
            f"foldseek lndb {queryDB_h} {queryDB_ss_h}",
            f"foldseek convert2fasta {queryDB_ss} {output_file_path}",
        ]

        for cmd in commands:
            print(f"Running:\n{cmd}")
            _run_command(cmd)

        print(f"Results written to {output_file_path}")


def from_seq_get_3di(input_dir, output_dir, args):
    """Run ProstT5 directory containing protein sequence fasta files, get 3di
    translation mode uses entire , or using CNN (mode: encoderOnly)"""
    device = get_torch_device(args.device)

    seq_files = list(input_dir.glob("*.fasta"))
    print(f"Input sequence files:\n{seq_files}")

    for seq_file in tqdm(seq_files):
        if args.translate:
            outfile_name = output_dir / f"{Path(seq_file).stem}_3di_translate.fasta"
            translate(
                in_path=seq_file,
                out_path=outfile_name,
                model_dir="Rostlab/ProstT5_fp16",
                split_char="!",
                id_field=0,
                half_precision=1,
                is_3Di=False,
                device=device,
                gen_kwargs=aa2ss,
            )
        if args.encoderOnly:
            outfile_name = output_dir / f"{Path(seq_file).stem}_3di_encoderOnly.fasta"
            get_embeddings(
                seq_path=seq_file,
                out_path=outfile_name,
                model_dir="Rostlab/ProstT5_fp16",
                device=device,
                split_char="!",
                id_field=0,
                half_precision=1,
                output_probs=False,
            )
        print(f"Results written to: {outfile_name}")


def create_arg_parser():
    """Create ArgumentParser"""
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

    if output_dir.is_dir():
        print(
            "WARNING: Result directory already exists; make sure to not overwrite results"
        )
    else:
        output_dir.mkdir()

    if mode == "struct":
        check_input(input_dir, "pdb")
        from_struct_get_3di(input_dir, output_dir)

    if mode == "seq":
        check_input(input_dir, "fasta")
        from_seq_get_3di(input_dir, output_dir, args)


if __name__ == "__main__":
    main()
