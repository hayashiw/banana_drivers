import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    sh = os.path.join(HERE, "singlestage_multithread_openblas.sh")
    os.execvp("bash", ["bash", sh, *argv])

if __name__ == "__main__":
    main()
