# DACL
> PANDORA DACL component

Installation / creating conda environment
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
└── utils
    └── thr_anom_labeler.py
```

Python modules
* dacl.py -- main entry point for DACL
* exec_mvi -- module for executing DACL MVI (missing value inference)
* exec_mli -- module for executing DACL MLI (missing label inference)
* exec_daug -- module for executing DACL DAUG (data augmentation)
* datatable -- module implementing DACL internal data representations
* reproducibility -- module ensuring reproducibility of all DACL experiments and evaluations
* cl_eval -- module implementing classification evaluation for DACL MLI and DAUG
* mvi -- package implementing DACL MVI algorithms
* mli -- package implementing DACL MLI algorithms
* daug -- package implementing DACL DAUG algorithms
* utils -- package providing utilities

  
