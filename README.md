
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
   conda env create -f environment.yml
   conda activate quantecon
   ```

If you would prefer to use a different environment but want to add the `tools` this can be done
via `pip`:

For building `jupyter-book`:

1. Jupyter Book: `pip install jupyter-book`
2. QuantEcon Theme: `pip install git+https://github.com/quantecon/quantecon-book-theme`

For conversion and publishing:

1. Jupytext: `pip install jupytext`
2. Github Pages Import Tool: `pip install ghp-import`


## Building Projects

To build the sandpit using `jupyter-book`:

1. To build the `lectures` collection you can use
   ```bash
   jb build lectures
   ```
   or
   ```bash
   jupyter-book build lectures
   ```
   This will by default save the outputs in `<lecture>/_build/html` within the project folder
2. If you are already in the `lectures` folder you can use:
   ```bash
   jb build ./
   ```
   or
   ```bash
   jupyter-book build ./
   ```
   This will by default save the outputs in `_build/html` within the project folder

There is currently no way to auto-launch the results after a build is complete.

To open the `html` you can use:

```bash
#OS X
open lectures/_build/html/index.html
#Linux (you could also use firefox etc.)
chrome lectures/_build/html/index.html
```

or browse to the folder via the finder.

A [QuantEcon suggested workflow](https://quantecon.github.io/QuantEcon.manual/writing/notebooks.html) is also available
for writing content via Notebooks.

