# BerMol dependency

ToxCPI and the tipFormer adapters require the BerMol Python package and
pretrained model for chemical feature extraction. They are not redistributed
in this package because the archived copy did not include a standalone
license. Obtain an authorized copy from the relevant upstream distribution,
then set `TOXCPI_BERMOL_DIR` to the directory containing the `bermol/` package.

## Pretrained model

The pretrained weight is deliberately not stored in this GitHub working tree:

- filename: `BerMolModel_base.pkl`
- size: `273643734` bytes
- SHA256: `afd1929fdbef6da110b5f4d0688a0f4a5f88fc3fb660bfcc1fb56e9624de38b1`

After obtaining the weight, set:

```bash
export TOXCPI_BERMOL_MODEL=/absolute/path/to/BerMolModel_base.pkl
```

`TOXCPI_BERMOL_DIR` can also be set to an alternative directory containing
the `bermol/` Python package.

## Redistribution note

No standalone license file was present in the archived BerMol source
directory. Therefore this package contains neither its Python source nor its
pretrained weight. `UPSTREAM_TIPFORMER_README.md` is retained for provenance
only.
