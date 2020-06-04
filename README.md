# COVID-19 Modelling

Work in progress state space model of the corona pandemic.

## Scripts

Setup

```bash
pip install -r requirements.txt
```

### C19.se

Download data from https://c19.se/ to a local JSON file:

```bash
python src/c19_se.py --output-file <name-of_data-file>.json
```

### TODO

- ~Setup CT test with tricky data, increased noise. Create situation where iterations are required.~
- ~Iterative version of SLR smoothing.~
- ~Naïve truncated sampling for FHM model(s)~
- Gauss sampling on subspace (states sum to 1)
- Toy SIR model
- Partly analytical SLR
