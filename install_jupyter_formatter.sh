conda install -c conda-forge black nodejs -y

jupyter labextension install @ryantam626/jupyterlab_code_formatter

conda install -c conda-forge jupyterlab_code_formatter -y

jupyter serverextension enable --py jupyterlab_code_formatter
