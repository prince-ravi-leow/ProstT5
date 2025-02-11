# 3Di-pipeline for `phylostruct` project on ku-01 machine
# Intro 
This is a unified pipeline that can extract 3Di from a directory containing either protein structures or amino acid sequences.

This is basically a [ProstT5 fork](https://github.com/prince-ravi-leow/ProstT5), configured such that it uses pre-downloaded model weights and foldseek binary already on the server.

* struct -> 3di : uses `foldseek` under the hood (called via `subprocess`) and disposes of the pesky temporary files
* aa_seq -> 3di : uses the `ProstT5` model under the hood – supports both the slower more accurate `translate` and faster less accurate `encoderOnly` mode

Read more in the [ProstT5 paper](https://academic.oup.com/nargab/article/6/4/lqae150/7901286)

# Installation
```sh
# Create conda environment
mamba create -n 3di-pipeline python=3.11 -y
mamba activate 3di-pipeline
pip install torch
pip install transformers
pip install sentencepiece
pip install protobuf

# git clone the repo
git clone https://github.com/prince-ravi-leow/ProstT5 
```

# Usage
* Run the `generate_3di.py` script, specify either `struct` or `seq` mode, depending on what your input type is.
* If you use `seq` mode please specify **EITHER** `--translate` or `--encoderOnly` mode, with the appropriate flag

```
python3 generate_3di.py -h
usage: generate_3di.py [-h] --mode {seq,struct} [--encoderOnly] [--translate]
                       [--device {cuda:0,cuda:1,mps,cpu}]
                       input_dir output_dir

    Generate 3di alphabet from either AA seq or prot struct, ONLY submit directory

    Requires ProstT5 + weights installation and foldseek


positional arguments:
  input_dir             Input sequences or structures
  output_dir            Output directory

options:
  -h, --help            show this help message and exit
  --mode {seq,struct}   Get 3Di alphabet from either aa SEQ or protein STRUCT
  --encoderOnly         Use 'encoderOnly' mode (FASTER, LESS ACCURATE)
  --translate           Use 'translate' mode (SLOWER, MORE ACCURATE)
  --device {cuda:0,cuda:1,mps,cpu}
                        Choose which device to run on (useful for distributing jobs across multiple GPU's)
```

# Limitation(s)
`encoderOnly` mode relies on a small couple megabytes CNN to run 

If `generate_3di.py` is run from the root of this repo, it will use the pre-downloaded `cnn_chkpnt/`, otherwise it will download it into the current working directory.