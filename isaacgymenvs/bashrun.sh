#!/bin/bash

while true
do
  python train.py task=activeperception num_envs=60
  echo "Running your command..."
  sleep 1 # Optional: to prevent overwhelming your system, add a delay between iterations
done

