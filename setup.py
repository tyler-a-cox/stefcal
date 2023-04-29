from setuptools import setup

import os
import sys
import json
from pathlib import Path

sys.path.append("hera_cal")


def package_files(package_dir, subdirectory):
    # walk the input package_dir/subdirectory
    # return a package_data list
    paths = []
    directory = os.path.join(package_dir, subdirectory)
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            path = path.replace(package_dir + '/', '')
            paths.append(os.path.join(path, filename))
    return paths


data_files = package_files('hera_cal', 'data') + package_files('hera_cal', 'calibrations')
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup_args = {
    'name': 'stefcal',
    'author': 'HERA Team',
    'url': 'https://github.com/tyler-a-cox/stefcal',
    'license': 'BSD',
    'description': 'collection of calibration routines to run on the HERA instrument.',
    'long_description': long_description,
    'long_description_content_type': 'text/markdown',
    'package_dir': {'stefcal': 'stefcal'},
    'packages': ['stefcal'],
    'include_package_data': True,
    'install_requires': [
        'numpy>=1.10',
        'scipy>=1.9.0',
        'pyuvdata @ git+https://github.com/RadioAstronomySoftwareGroup/pyuvdata@cal-init-method',
        "jax",
        "jaxlib",
    ],
    'extras_require': {
        "all": [
            'optax'
        ],
        'dev': [
            "pytest",
            "pre-commit",
            "pytest-cov",
            "hera_sim",
            "pytest-xdist"
        ]
    },
    'zip_safe': False,
}


if __name__ == '__main__':
    setup(*(), **setup_args)
