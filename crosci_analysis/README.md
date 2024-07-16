# CROSCI package

## 1.Installation

First, create, and activate nbt2 conda environment:

```conda create -n crosci python=3.11```

```conda activate crosci```

Install compiler:
- On mac (from terminal):
```
xcode-select --install
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install llvm
```
- On linux: ```conda install gcc```
- On windows: go to https://visualstudio.microsoft.com/visual-cpp-build-tools/ and install C++ tools.

Then, with the crosci environment active go to the directory that contains crosci (e.g. if crosci is at the path /home/user/some_directory/crosci , you should ```cd /home/user/some_directory```), and execute: 
```
pip install -r crosci/requirements.txt
```

Now we need to prepare the environment for compilation. This is applicable only on mac. For windows/linux there is nothing needed.
```
export SDKROOT=$(xcrun --sdk macosx --show-sdk-path)
export LIBRARY_PATH="$LIBRARY_PATH:$SDKROOT/usr/lib"
export PATH="/usr/local/opt/llvm/bin:$PATH"
```

After this, still in the crosci directory, compile the c code required for running DFA and fEI:
```python crosci/setup.py build_ext --inplace```

## 2.Running (demo)

The demo script contains randomly generated data for which DFA, fEI and bistability are computed. You can use it as a starting point for your own data.
```commandline
python crosci/demo.py
```

If you would like to keep the demo script separate from the directory where the crosci code is located, you need to set the variable crosci_code_path in demo.py to the path where crosci is located.