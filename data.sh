# Download the Centrum Wiskunde & Informatica (CWI) Walnut Dataset
DATADIR=data
mkdir $DATADIR

zenodo_get https://doi.org/10.5281/zenodo.2686725 -o $DATADIR
zenodo_get https://doi.org/10.5281/zenodo.2686970 -o $DATADIR
zenodo_get https://doi.org/10.5281/zenodo.2687386 -o $DATADIR
zenodo_get https://doi.org/10.5281/zenodo.2687634 -o $DATADIR
zenodo_get https://doi.org/10.5281/zenodo.2687896 -o $DATADIR
zenodo_get https://doi.org/10.5281/zenodo.2688111 -o $DATADIR

unzip "$DATADIR/*.zip" -d $DATADIR

# Construct ground truth
srun python utils/construct_ground_truth.py -d $DATADIR
