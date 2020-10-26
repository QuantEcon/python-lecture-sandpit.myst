
# Lectures in Quantitative Economics Test Site (Myst)

This is a sandpit for the Python Lectures

## Setup

Getting this repository setup:

1. Download and install [Anacoda](https://www.anaconda.com/distribution/) python distribution
2. Fetch this repository from `git`
   ```bash
   git clone https://github.com/QuantEcon/python-lecture-sandpit.myst
   cd python-lecture-sandpit.myst
   ```
3. Install the pre-configured `conda` environment
   ```bash
   conda create -f environment.yml
   conda activate quantecon-sandpit
   ```

## Building

There are two environments setup that we can use to develop lectures:

1. `lectures` -> using `jupyter` notebooks
2. `lectures-myst` -> using `myst markdown` files

To build the sandpit using `jupyter-book`:

1. To build the `lectures` collection you can use
   ```bash
   jb build lectures
   ```
   or
   ```bash
   jupyter-book build lectures
   ```
2. To build the `lectures-myst` collection you can use:
   ```bash
   jb build lectures-myst
   ```

This will by default save the outputs in `_build/html` within each folder.

There is current no way to auto-launch the results after a build is complete.

To open the `html` you can use:

```bash
open lectures/_build/html/index.html    #OS X
chrome lectures/_build/html/index.html  #Linux (you could also use firefox etc.)
```

