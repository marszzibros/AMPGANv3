#!/bin/bash

for m in {1..30}; do
  sbatch generate.sh "$m"
done
