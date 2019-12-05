#!/bin/bash

for proc in {4..16..4}
do
    mpirun -machinefile ~/machines-openmpi --bynode -np $proc /tmp/node000-bccd/jacobi
done