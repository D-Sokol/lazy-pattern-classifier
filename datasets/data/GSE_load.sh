#!/bin/bash

cd "$(dirname "$0")"
wget 'https://sbcb.inf.ufrgs.br/data/cumida/Genes/Liver/GSE14520_U133A/Liver_GSE14520_U133A.csv'
wget 'https://sbcb.inf.ufrgs.br/data/cumida/Genes/Breast/GSE70947/Breast_GSE70947.csv'
wget 'https://sbcb.inf.ufrgs.br/data/cumida/Genes/Colorectal/GSE44076/Colorectal_GSE44076.csv'
wget 'https://sbcb.inf.ufrgs.br/data/cumida/Genes/Prostate/GSE6919_U95B/Prostate_GSE6919_U95B.csv'
