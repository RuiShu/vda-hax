import subprocess
import os

def main():
    if not os.path.exists('data'):
        os.makedirs('data')

    if os.path.exists('data/mnist.npz'):
        print "Using existing data"
    else:
        print "Opening subprocess to download data from URL"
        subprocess.check_output(
            '''
            cd data
            wget https://s3.amazonaws.com/img-datasets/mnist.npz
            ''',
            shell=True)

if __name__ == '__main__':
    main()
