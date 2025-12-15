#!/bin/bash
set -e

eval "$(conda shell.bash hook)"
conda create -n hora python=3.8 -y
conda activate hora

pip install -r requirements.txt

wget "https://drive.usercontent.google.com/download?id=1StaRl_hzYFYbJegQcyT7-yjgutc6C7F9&export=download&confirm=t" -O isaac4.tar.gz
tar -xzf isaac4.tar.gz
pip install -e ./isaacgym/python
rm isaac4.tar.gz

wget "https://drive.usercontent.google.com/download?id=1xqmCDCiZjl2N7ndGsS_ZvnpViU7PH7a3&export=download&confirm=t" -O data.zip
unzip data.zip -d cache/
rm data.zip

wget "https://drive.usercontent.google.com/download?id=17fr40KQcUyFXz4W1ejuLTzRqP-Qu9EPS&export=download&confirm=t" -O data.zip
unzip data.zip -d outputs/AllegroHandHora/
rm data.zip

ENV_PATH="$CONDA_PREFIX"

mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
mkdir -p "$CONDA_PREFIX/etc/conda/deactivate.d"

cat > "$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh" <<EOL
#!/bin/sh
export OLD_LD_LIBRARY_PATH="\$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH"
EOL

cat > "$CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh" <<EOL
#!/bin/sh
export LD_LIBRARY_PATH="\$OLD_LD_LIBRARY_PATH"
unset OLD_LD_LIBRARY_PATH
EOL

chmod +x "$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh"
chmod +x "$CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh"
