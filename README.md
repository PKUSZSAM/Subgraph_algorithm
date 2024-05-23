# Subgraph Isomorphism Algorithm
This repo includes samples codes of the subgraph generation, subgraph isomorphism matching, subgraph GED verification.

This repository is adapted from the codebase used to produce the results in the paper "Rapid mining of fast ion conductors via subgraph isomorphism matching".

## Requirements

The installation can be done quickly with the following statement.
```
pip install -r requirements.txt
```

We recommend using the Anaconda Python distribution, which is available for Windows, MacOS, and Linux. Installation for all required packages (listed above) has been tested using the standard instructions from the providers of each package.

## Data

The non-Li framework structures of 104 candidates are at
```
./data/structure/
```

Examples of the subgraph representation of four structure types and prototype with graph distance of four are at
```
./data/structure_type
```

## Method quick start

Here we use the structure format of atom.config, which is the standard input format of PWmat code.

1. subgraph generation using area distance pair clustering
```bash
python subgraph_generation.py --path /where/your/atomconfig --gdist 3 --nbr_type voronoi_area
```

2. subgraph matching
```bash
python subgraph_matching.py --struct1 /where/your/atomconfig --struct2 /where/your/atomconfig
```

3. subgraph graph edit distance (GED) verification
```bash
python subgraph_GED_verification.py --subgraph1 /where/your/gmlfile --subgraph2 /where/your/gmlfile 
```


