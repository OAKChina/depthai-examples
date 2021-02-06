# coding=utf-8
def install_dep():
    import subprocess
    import sys

    # https://stackoverflow.com/a/58026969/5494277
    in_venv = (
        getattr(sys, "real_prefix", getattr(sys, "base_prefix", sys.prefix))
        != sys.prefix
    )
    pip_call = [sys.executable, "-m", "pip"]
    pip_install = pip_call + ["install"]
    if not in_venv:
        pip_install.append("--user")
    try:
        subprocess.check_call([*pip_install, "pip", "-U"])
        # temporary workaroud for issue between main and develop
        # subprocess.check_call([*pip_call, "uninstall", "depthai", "--yes"])
        subprocess.check_call([*pip_install, "-r", "../requirements.txt"])
    except subprocess.CalledProcessError as ex:
        print(
            f"Optional dependencies were not installed (exit code {ex.returncode})"
        )


try:
    from .depthai_0021 import *
except ImportError:
    install_dep()

# flag = True
# v = f"depthai=={depthai.__version__}"
#
# with open("../requirements.txt", "a+", encoding="utf8") as f:
#     for i in f.readlines():
#         print(i)
#         print('*' * 40)
#         if v == i:
#             flag = False
# print(flag)
# if flag:
#     install_dep()
