
# IA & BD

pwd wifi: CiclesM@ster

## Iniciar Jupyter Notebook
```bash
conda activate entorn_1_Jaume
```
```bash
jupyter notebook --no-browser --port 8081
```

```bash
http://localhost:8081/tree?
```

## Instal·lació de mini conda

Descarregar mini conda de la web, per comprobar que l'haguem instal·lat:

```bash
conda -v
```

## Creació del entorn virtual
```bash
conda create--name nom
```

```bash
conda info --envs
```

Per poder accedir al ``Notebook`` hem d'activar l'entorn virtual que acabem de crear.

```bash
conda activate nom
```

## Instal·lar Paquets
```bash
conda install ipython
conda install numpy
conda install matplotlib
```

## Instal·lar Jupyter Notebook
```bash
conda install -c anaconda jupyter
```

## Arrancar jupyter
```bash
jupyter notebook --no-browser --port 8081
```
- Desde el navegador accedim a aquest port.
- Si ens demana un token, sortirà a la consola.

## Comandes bàsiques:
- esc + 

| Acció                 | Tecla        | 
|-----------------------|--------------|
| Add cell below        | b            | 
| Add cell above        | a            |
| Delete current cell   | dd           |
|                       |              |
| Code cell             | y            |
| Text cell             | m            |
| Desactivate code cell | r            |
| Open Info             | h            |
| Veure snippets metode | shit + tab   |
|                       |              |
| Copy cell             | c            |
| Paste cell            | v            |
|                       |              |
| Run cell              | ctrl + enter |

