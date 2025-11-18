import platform
import subprocess
import sys
from pathlib import Path


def main():
    outdir = Path("outputs")
    outdir.mkdir(exist_ok=True)
    with open(outdir / "env_info.txt", "w") as f:
        f.write("OS: " + platform.platform() + "\n")
        f.write("Python: " + sys.version.replace("\n", " ") + "\n")
        # pip list filtered
        try:
            pkgs = subprocess.check_output([sys.executable, "-m", "pip", "list"]).decode()
        except Exception:
            pkgs = "<pip list failed>\n"
        f.write("\n[pip list]\n")
        f.write(pkgs)
        # versions of key libs
        def pip_show(name):
            try:
                return subprocess.check_output([sys.executable, "-m", "pip", "show", name]).decode()
            except Exception:
                return f"<{name} not found>\n"
        f.write("\n[aif360]\n" + pip_show("aif360"))
        f.write("\n[fairlearn]\n" + pip_show("fairlearn"))
        f.write("\n[witwidget]\n" + pip_show("witwidget"))
        f.write("\n[tensorflow]\n" + pip_show("tensorflow"))
        f.write("\n[tfma]\n" + pip_show("tensorflow-model-analysis"))


if __name__ == "__main__":
    main()
