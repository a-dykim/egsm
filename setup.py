from setuptools import setup
import glob
import os
import sys


def package_files(package_dir,subdirectory):
    # walk the input package_dir/subdirectory
    # return a package_data list
    paths = []
    directory = os.path.join(package_dir, subdirectory)
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            path = path.replace(package_dir + '/', '')
            paths.append(os.path.join(path, filename))
    return paths

data_files = package_files('egsm','data')

setup_args = {
    'name': 'egsm',
    'author': 'Doyeon Avery Kim',
    'url': 'https://github.com/a-dykim/egsm', ## change
    'license': 'BSD',
    'description': 'extended Global Sky Model',
    'package_dir': {'egsm': 'egsm'},
    'packages': ['egsm'],
    'include_package_data': True,
    'scripts': [],
    'package_data': {'egsm': data_files},
    'install_requires': ['NonnegMFPy'],
    'zip_safe': False,
}


if __name__ == '__main__':
    setup(**setup_args)
