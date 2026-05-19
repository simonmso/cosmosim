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
