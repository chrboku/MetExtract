# MetExtract II

A comprehensive metabolite extraction and analysis tool for stable isotope labeled, untargeted metabolomics. 

## Installation

### Install uv
Please install uv. For instruction see [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/).

### Install git
Please install git. For instructions use [https://git-scm.com/](https://git-scm.com/). 

### Clone MetExtract II
```bash
# Clone the repository
git clone https://github.com/chrboku/MetExtract-II
cd MetExtract-II
```

## Module selection
MetExtract consists of two modules, namely ```TracExtract``` and ```AllExtract```. TracExtract ist designed for finding metabolism products of endogenous or exogenous secondary metabolites that are not participating in the primary metabolism (e.g., glucose), while AllExtract is designed to find completely isotopically labeled metabolites from co-cultivation experiments. 

## Quick Start
```bash
uv run python -m src.MExtract --module MODULE
```

## Usage

### Running the Applications

```bash
# Main application (MExtract) - Primary metabolite extraction interface
uv run python -m src.MExtract --module MODULE

# MetExtract II Main Interface - Alternative main interface
uv run python -m src.MetExtractII_Main
```


### Code Formatting
MetExtract uses the ruff code formatter that can be executed with the following commands.
```bash
# Format code with Black
uvx ruff format
```

# License
This project is licensed under the GNU GENERAL PUBLIC LICENSE (see LICENSE.txt for a full copy of the licensing conditions).