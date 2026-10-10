# DACL
> PANDORA DACL component

CONDA ENVIRONMENT CREATION:
```sh
conda env create -f environment.yml
conda activate dacl_env
pip install -r requirements.txt
```

Source code tree
```text
code/
├── cl_eval.py
├── dacl.py
├── datatable.py
├── daug
│   ├── daug.py
│   ├── daug_impl.py
│   └── tabpfgen_local.py
├── exec_daug.py
├── exec_mli.py
├── exec_mvi.py
├── mli
│   └── mli.py
├── mvi
│   ├── imputer.py
│   ├── mvi_statistic.py
│   ├── mvinn.py
│   ├── mvinn_sw.py
│   ├── odeval.py
│   ├── regeval.py
│   └── zeroimputer.py
├── reproducibility.py
```

└── utils
    └── thr_anom_labeler.py
