import sys
from .server import run_server

if __name__ == '__main__':

    ppath = sys.argv[1]
    try:
        portArg = sys.argv[2]
    except:
        portArg = None

    print(f"Supplied pickle_path is: {ppath}")

    vizVars = {}

    if portArg is None:
        run_server(ppath, vizVars=vizVars)
    else:
        run_server(ppath, vizVars=vizVars,  PORT=portArg)

