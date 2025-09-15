from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("rfbid")
except PackageNotFoundError:
    __version__ = "0.1.0"  # fallback for dev installs
