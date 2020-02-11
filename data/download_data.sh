#!/usr/bin/env bash

if [[ ! -f ~/Medley-solos-DB.tar.gz ]]; then
    echo "Medley-solos-DB.tar.gz... Hold your horses, this might take a while!"
    wget https://zenodo.org/record/3464194/files/Medley-solos-DB.tar.gz?download=1
    mv Medley-solos-DB.tar.gz?download=1 ~/Medley-solos-DB.tar.gz
    tar -zxvf ~/Medley-solos-DB.tar.gz ~/Medley-solos-DB/
else
    echo "Medley-solos-DB.tar.gz already exists"
fi