#!/bin/bash

srun -u -n1 --gres=gpu:1 build/cugemv