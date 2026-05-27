# AST5220 Final project
I used this project as an opportunity to learn Cython. Good luck getting it to run on your own machine (seriously, feel free to email me if it doesn't work).

## Installation
Installation is, in theory, as simple as cloning this repository and running
```zsh
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```


### CyRK Installation troubles
You may need to install `libomp` if you don't already have it. I did, and then recompiled CyRK (the ODE solver for cython) with

```zsh
brew install libomp
pip install --no-cache-dir --force-reinstall CyRK  
```

On import, `CyRK` was unable to find omp. This I fixed this with

```zsh
export DYLD_LIBRARY_PATH="$(brew --prefix libomp)/lib:$DYLD_LIBRARY_PATH"
```

Finally, I could test my installation in python with
```py
from CyRK import test_pysolver, test_cysolver, test_nbrk
test_cysolver()
# Should say: CyRK's CySolver was tested successfully.
```

## Building the code

Compile the cython to c with
```zsh
python3 setup.py build_ext --inplace
```

## Generate figures
The actual code for generating figures is split into a file or two for each milestone. They are in `./scripts` and should be run from the parent directory with

```zsh
python3 -m scripts.ms1_data -d ./data
python3 -m scripts.ms1_plot -d ./data -o ./results/milestone1/
python3 -m scripts.ms2 -o ./results/milestone2
python3 -m scripts.ms3 -o ./results/milestone3
python3 -m scripts.ms4_data -o ./results/milestone3
python3 -m scripts.ms4_plot -o ./results/milestone3
```

where the scripts expect `./data` to have `supernovadata.txt`, `planck_cell_low.txt`, and `COM_PowerSpect_CMB-TT-binned_R3.01.txt`. 