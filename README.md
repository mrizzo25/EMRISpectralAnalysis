# EMRISpectralAnalysis


# Kludge WF installation 

The particular version of the EMRI Kludge suite which outputs plus and cross strain polarization amplitudes can be retrieved from my github. To clone, use:

`git clone --branch spectra_output https://github.com/mrizzo25/EMRI_Kludge_Suite.git`

NB: the spectra generation code will still work with the normal AAK output, bearing in mind that the output strains will be h\_I and h\_II, as defined in the AAK paper, with or without the detector response. The frequency content will be the same.

Make sure to set the environment variable `EKS_PATH`, which can be achieved by putting the following in your `~/.bashrc` file:

`export EKS_PATH="path to kludge suite"`

# Running Spectra Code
