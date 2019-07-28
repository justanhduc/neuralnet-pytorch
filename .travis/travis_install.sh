#!/usr/bin/env bash
if test -e $HOME/miniconda2/envs/pyenv; then
    echo "pyenv already exists."
else
    echo "Creating pyenv."
    if [[ $TRAVIS_PYTHON_VERSION == '3.4' ]]; then conda create --yes -q -n pyenv python=3.4 ; fi
    if [[ $TRAVIS_PYTHON_VERSION == '3.5' ]]; then conda create --yes -q -n pyenv python=3.5 ; fi
    if [[ $TRAVIS_PYTHON_VERSION == '3.6' ]]; then conda create --yes -q -n pyenv python=3.6 ; fi
    if [[ $TRAVIS_PYTHON_VERSION == '3.7' ]]; then conda create --yes -q -n pyenv python=3.7 ; fi
fi

source activate pyenv
