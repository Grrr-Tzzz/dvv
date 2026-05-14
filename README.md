This is my first time using github and writing something formal, so please bear with me.

# Environment

In case you cannot run this code, try to import this environment I used:

environment.yml

by "conda env create -f environment.yml", you may want to change the name of this env, for it is very strange.

# Compute_dvv
## Usage

python ./compute_dvv.py \
  --config THE_CONFIG_FILE.yaml \
  --no-save-csv \ if you do not want to save the result into csv
  --plot \ if you want to save the plot as png
  --rm \ whether to delete the last result
  --skip \ whether to skip the correlation. Usually use this when the correlation result is already saved.

you can also override the default value of save-csv, plot, rm and skip in config.yaml.

## Config.yaml
This is an example config that resembles the format of that paper.
