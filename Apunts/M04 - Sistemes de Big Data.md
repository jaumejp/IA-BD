# M04 - Sistemes de Big Data

Obtencio de ades &rarr; Curació de dades (canvi unitats) &rarr; Visualització &rarr; Extracció de conclusions

Tenim dades publiques o privades per diferents tipus de dades: 
- Relacionals
- No relacionals 
- No estructurades (fitxers independents com correus, webs, etc)

### Projectes big data (SI / NO)?

Per definir un projecte fem servir la regla de les 5V's
- Volum (quantitat)
- Velocitat (generació de dades)
- Varietat (dades de diferents fonts)
- Veracitat (dades reals)
- Variabilitat (dades diferents)

### No necessiten big data:

Botiga de llibres fets servir amb col·lecció modesta de llibres i atenen pocs clients (ho tenen en una llibreta)
- no té volum de dades
- no aportem valor
- no té varietat de dades (nomès té un sql) relacional i no relacional (clau valor, documents)

Un blog personal de vacanses on una persona publica coses
- no té velocitat de generació i actualització

Un petit restaurant local i aten a poques persones
- no té veracitat a les dades

---
---

# Curs de Big Data desde 0

## 1. Antecedentes a Big Data

### La informática tradicional: De la calculadora a la computación distribuida

La base del big data es la computacio

Informátrica tradicional: Calculadora, Àbac.

Dins de tota la informació, big data, és tractar dades

Recolectar > Emgagatzemar > Procesar > Visualitzar

Ex: calculadora

Introduim dades, memoria interna que els guarda, procesa les operacions i mostra les dades.

Ex: Analitzar factures
Factures dies anteriors, Programa informatic de contabilitat, calcular dades, informes o indicadors

Tractament de dades: treure partit a les dades

Si les dades són molt grans, la calculadores no donen a l'abast -> Augmentar màquines (creixement vertical). També tenim creixement horitzontal, posar més maquines petites (exemple de posar més cavalls al carro), l'altre exemple sería fer el cavall més fort. 

La computació distribuida, basicament ve a resoldre el problema del gran volum de dades i com procesar-los ja que en una màquina petita no és possible.

### Limitacions de la computació distribuida

En les no distribuides -> és lineal (ja que nomès ho fa una)

Les dades amb brut, s'han de partir, per tractar-les per separat.

Cada màquina té un fragment.

Analitzar cada conjunt.

Tornar-ho a unificar per procesar tots els resultats junts per treure un resultat final. 

Partir, procesar, transportar i ajuntar

### La clave del Big Data: Localización del dato

Quan van creixen les dades, el escalat horitzontal no ens serveix perquè tenim limitacions de temps. 

Big data -> va veure que el gran problema era moure les dades cap a aquestes maquines (transferir-ho per la xarxa).

No ho fem, el procesaent es fa a cada màquina. Procesem on estàn les dades, no les enviem a un servidor central.

## 2. Entender que és Big Data

### Big Data no es magia, son 3Vs

1. Volumen (datos hasta el infinito)
2. Velocitat (apareix el concepte streaming)
3. Variabilidad (cualsevol casa es transforma en dades utilitzables)

### Volumen: Datos hasta el infinito

TeraByes, PetaBytes, ExaBytes...

Emgagatzemen en servidors infinits també perquè clusters de fins a 10.000 màquines que treballen a la vegada.

### Velocidad: Aparece el concepto de Streaming

Streaming -> procesar según llega el dato.
Nunca és real time, siempre és "Near Real Time"

### Variabilidad: Cualquier cosa se transforma en dato utilizable

Analitzem cualsevol dada (estructurada, semi-estructurada, des-estructurada).

Tot element que generi dades és útil per big data.

### Hay más V's, pero compuestas de estas tres

Veracidad (han de ser reals)
Valor (necesitem aportar valor)
Viabilidad (dades accesibles)
Visualización (si no les sabem tractar, no serveixen)

## 3. Big data y negocio

### Nuevas oportunidades de negocio

La imaginació es el que ens limita. Dades de xarxes socials, de sensors, etc.

Si el producto es gratis, el negocio eres tu.

Los datos són el nuevo oro. Més dades dels clients, millors conclusions

### Nuevas oportunidades laborales

Noves tecnologies -> Noves oportunitats laborals, es crea un nou mercat.

### Perfiles técnicos de trabajo en Big Data

Analista de dades / Data Science

Desarrollador / Data Engineer

Administrador / Big Data Sysadmin

## 4. Plataforma Big Data

### ¿Qué es un Cluster Hadoop?

Conjunto de servidores que trabajan coordinados

Ordinador més potent és un: Servidor i un Cluster és conjunt de servidors

### Tipos de servidores

Servidors maestros -> coordinar el cluster
Trabajadores -> alojan y procesan los datos
Ingesta -> conexión del cluster con el exterior
Utilidades -> Servicios adicionales

### ¿Qué se considera una Plataforma?

Cluster Hadoop es el core de una Plataforma Big Data, però hay más cosas

### Capas de una Plataforma Big Data

## 5. Hadoop: Core tecnológico de Big Data

### Apache Hadoop: Core tecnológico de Big Data

### Historia de Apache Hadoop

### HDFS y YARN: Almacenamiento y procesamiento como punto central

### Ecosistemas Hadoop: Diferentes productos para diferentes finalidades

### Ubicar los componentes en sus capas correspondientes

### Hadoop es al Big Data lo que Linux a los Sistemas Operativos

## 6. Proveedores de productos Big Data

### Plataformas Apache Hadoop: Cloudera y Hortonworks

### Bases de datos NoSQL

### Cloud Soluciones Hadoop, productos PaaS

## 7. Ejemplos de aplicación de Big Data

### Industria: Mejora de tiempos en procesos de larga duración

### Operadora TV: Análisis de la calidad de señal de televisión

### Debate político: Conocer en tiempo real la respuesta de las RRSS