#!/usr/bin/env bash

inp=${1}
outp=${2}

parc="aal des hox cc2"
set -x

for dir in `ls ${inp}`;
do
  inp1=${inp}/${dir}
  for a in $parc;
  do
    python /ocean/projects/med220004p/gkiar/code/gkiar/cpe/code/compute_similarity.py ${inp1} ${outp} --atlas ${a}
  done
done
