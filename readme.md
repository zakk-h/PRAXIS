# LicketyRESPLIT: Fast Rashomon Set Approximation for Sparse Decision Trees

LicketyRESPLIT is a high-performance C++/Python library for rapidly enumerating all decision trees within a multiplier of the best tree (or reference solution).

LicketyRESPLIT can generate millions of valid trees in seconds, orders of magnitude faster than other approaches.

## Installation

### Building from Source

If you have access to C++ build tools, you can install directly from GitHub:

```bash
pip install "git+https://github.com/zakk-h/LicketyRESPLIT.git@praxis"
```

### Prebuilt Binary Wheels

To make installation simpler, LicketyRESPLIT provides prebuilt binary wheels for all major platforms:

- **Windows** (64-bit)
- **macOS** (Intel and Apple Silicon)
- **Linux** (any distribution)

**Supported Python versions:** 3.9 – 3.13

#### Installation Steps

1. Go to the release page: https://github.com/zakk-h/LicketyRESPLIT/releases/tag/praxis-v0.0.8

2. Choose the wheel that matches:
   - Your Python version (cp39, cp310, cp311, cp312, cp313)
   - Your operating system

3. Install it with pip (replace the URL with your chosen wheel file from step 2):

**macOS (Apple Silicon), Python 3.11:**
```bash
pip install "https://github.com/zakk-h/LicketyRESPLIT/releases/download/praxis-v0.0.8/praxis-0.0.8-cp311-cp311-macosx_11_0_arm64.whl"
```

## Example

See [`examples/example.ipynb`](examples/example.ipynb) for a complete walkthrough of using PRAXIS.