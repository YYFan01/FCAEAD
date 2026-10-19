# FCAEAD

This directory contains the core Python implementation of FCAEAD and the
illustrative dataset used in the supplementary material of the paper:

> "A novel uncertainty
> measure integrating fuzzy beta-covering rough set and information entropy
> for data anomaly detection," submitted to *IEEE Transactions on Fuzzy
> Systems*.

## Citation requirement

**If you find this code useful in your research, please consider citing the
associated paper. Once the paper is published, please refer to the complete
bibliographic information and DOI of the published article.

## Files

- `FCAEAD.py`: FCAEAD anomaly-score calculation and the executable example.
- `compute_co_Neighborhood.py`: construction of the fuzzy beta-covering
  relation matrix.
- `compute_Function.py`: the four overlap/grouping functions used by FCAEAD.
- `Example.xlsx`: the illustrative dataset used in the supplementary material.
- `requirements.txt`: Python package dependencies.

## Environment

The code was tested with Python 3.10.19. Install the required packages with:

```bash
pip install -r requirements.txt
```

## Usage

Run the example from this directory:

```bash
python FCAEAD.py
```

Expected output:

```text
[1.2172468 1.2362254 1.199481  1.2211401 1.3319135]
```

The main interface is:

```python
scores = FCAEAD(data, beta, function, p)
```

where `data` is a two-dimensional NumPy array, `beta` is the fuzzy
beta-covering parameter, `function` is one of `O_mp`, `O_p`, `G_Mp`, and
`G_p`, and `p` corresponds to the function parameter $\mu$ in the paper.