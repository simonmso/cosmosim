### CyRK Installation troubles
Things that maybe worked for me
```zsh
brew install libomp
pip install --no-cache-dir --force-reinstall CyRK  
```

On import, `CyRK` was unable to find omp. This I fixed with

```zsh
export DYLD_LIBRARY_PATH="$(brew --prefix libomp)/lib:$DYLD_LIBRARY_PATH"
```

Finally, I could test my installation in python with
```py
from CyRK import test_pysolver, test_cysolver, test_nbrk
test_cysolver()
# Should say: CyRK's CySolver was tested successfully.
```

### Building the code

```zsh
python3 setup.py build_ext --inplace
```

### Running the code

```zsh
python3 -m scripts.ms1_data -d ./data
python3 -m scripts.ms1_plot -d ./data -o ./results/milestone1/
python3 -m scripts.ms2 -o ./results/milestone2
python3 -m scripts.ms3 -o ./results/milestone3

```