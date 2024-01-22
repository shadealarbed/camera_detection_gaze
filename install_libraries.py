# File: install_libraries.py

import subprocess

def install_libraries():
    libraries = ['opencv-python', 'dlib', 'numpy','cmake']

    for lib in libraries:
        try:
            subprocess.check_call(['pip', 'install', lib])
            print(f'Successfully installed {lib}')
        except subprocess.CalledProcessError:
            print(f'Failed to install {lib}')

if __name__ == "__main__":
    install_libraries()
