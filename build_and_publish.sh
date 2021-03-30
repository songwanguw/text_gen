#!/bin/bash

#This code is used to build and publish PySummarization package
#pip install --user --upgrade setuptools wheel twine gitpython

dirty_dirs=("build" "dist" "*.egg-info")

for x in ${dirty_dirs[@]}; do
	[ -e $x ] && rm -rf $x
done

pip install twine keyring artifacts-keyring
pip install --user --upgrade setuptools wheel twine gitpython

python setup.py  bdist_wheel
