import sys
from .server import run_server

if __name__ == '__main__':

    ppath = sys.argv[1]
    try:
        portArg = sys.argv[2]
    except:
        portArg = None

    print(f"Supplied pickle_path is: {ppath}")

    if portArg is None:
        run_server(ppath)
    else:
        run_server(ppath, PORT=portArg)

