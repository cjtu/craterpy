#!/bin/bash

if [[ $TRAVIS_OS_NAME == 'osx' ]]; then

    # update stuff
    case "${TOXENV}" in
        py27)
            wget https://repo.continuum.io/miniconda/Miniconda2-latest-MacOSX-x86_64.sh
            ;;
        py34)
            wget https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
            ;;
    esac
else
    if [[ "$TRAVIS_PYTHON_VERSION" == "2.7" ]]; then
      wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh -O miniconda.sh
    else
      wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    fi
fi
