Bootstrap: docker

From: continuumio/miniconda3

%files
    environment.yml

%post
    /opt/conda/bin/conda env create -f environment.yml

%runscript
    exec /opt/conda/envs/$(head -n 1 environment.yml | cut -f 2 -d ' ')/bin/"$@"
