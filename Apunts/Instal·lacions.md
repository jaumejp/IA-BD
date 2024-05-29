## Instal·lació WSL

Obrir powershell (com a administrador)

WSL (requereix windows 10.2. i posterior).

Instal·lar ubuntu 20.04

wsl --list --online

wsl --install --d Ubuntu-20.04

Tancar powershell i reiniciar windows

Icona WSL

## Instal·lació miniconda

https://kontext.tech/article/1064/install-miniconda-and-anaconda-on-wsl-2-or-linux

sudo apt-get update

sudo apt-get install wget

wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.12.0-Linux-x86_64.sh


sh ./Miniconda3-py39_4.12.0-Linux-x86_64.sh

Enter

yes > acceptar la llicència.

Enter

yes > iniciar el miniconda

Acaba amb Thank you for installing Miniconda3!


Al fitxer ~/.bashrc hi ha paràmetres per executar el conda.

## Creació de l'entorn

conda create --name pia --python 3.9

conda create --name pia

conda activate pia > entrem a l'entorn pia.  

conda info --envs

conda info > Canals: URLs amb repositoris de mòduls.

conda install ipython       > intèrpret de comandes

conda install jupyter -c anaconda > a través del canal anaconda (no els estàndard)

## Accés al Jupyter Notebook

Crear la carpeta $HOME/pia

Copiar-hi el fitxer  nb01_descriptive_statistics.ipynb (del moodle)

Accés des del Windows:  \\WSL$   (o via recursos compartits de xarxa)

  
A l'Ubuntu:  
jupyter notebook >> això l’obriria al navegador i no en tenim.  

jupyter notebook --no-browser --port=8081

això dona un token => agafar-lo

  
Al Windows:  

Navegador:    localhost:8081

podem copiar el token com a password o el posem a sota i posem un pwd més fàcil.

Hi ha un arbre de directoris > accedim al notebook.

-----

Veiem que volem fer servir numpy i matplotlib => tornem a l'Ubuntu.

conda activate pia

conda install numpy

conda install matplotlib -c conda-forge

Al navegador, Kernel > restart