# Introduction
This repository contains Code for my bachelors thesis "**Untersuchung und Vergleich von Machine Learning Methoden zur synthetischen Erzeugung von Finanzdaten**".

## Existing Implementations
Already existing implementations of the following models were used and evaluated:
- [TimeGAN](https://github.com/jsyoon0823/TimeGAN)
- [QuantGAN](https://github.com/JamesSullivan/temporalCN)

## Misc
### Explanation on how to copy other repos inside this repo:
- `git clone <url_of_repo_to_clone>` *clone repo*
- `cd <path_to_clone_repo_to>` *navigate to folder where repo should be cloned to*
- `cp -r <root_folder_of_repo_to_clone>/* .` *clone repo to current directory*
    - cp - copy
    - -r all files recursively
    - /* makes sure all files are copied from within the folder 
    - . copies files to current directory
- `rm -rf <path_to_repo_to_clone>` *remove original cloned repo*
