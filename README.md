# Clasificación de Expresiones Genómicas

## Introducción
Dos grupos de investigación del ![CONICET](https://www.conicet.gov.ar/) han unido esfuerzos para combinar tecnologías de secuenciación de nueva generación con inteligencia artificial. Su objetivo es analizar mediciones de ARN de 200 genes recolectados de pacientes con lesiones pre-tumorales, con la finalidad de comprender cómo las células en estado de hiperplasia pueden evolucionar hacia tumores malignos. Para lograrlo, llevaron a cabo un estudio utilizando muestras de hiperplasias de pacientes que fueron monitoreados durante 5 años, con el fin de identificar la transformación hacia neoplasias o hiperplasias más agresivas. Estas muestras se almacenaron en un biobanco y se etiquetaron según el pronóstico observado.

<img src='/imagenes/cancer.jpg' width='500' height='350'>


## Objetivo
Nuestro estudio tiene como objetivo determinar los factores genéticos que influyen en el pronóstico de pacientes con lesiones pre-tumorales. Analizaremos muestras que han sido etiquetadas como de buen o mal pronóstico según la evolución clínica observada durante el seguimiento de 5 años. Al identificar los perfiles genéticos asociados con estos dos grupos de pronóstico, buscamos mejorar la detección y el tratamiento temprano de pacientes en riesgo de desarrollar cáncer a partir de lesiones pre-tumorales.

##  Separación de los Datos
Decidimos dividir nuestros datos en dos conjuntos: uno para desarrollo y otro para validación. Dado que contamos con solo 500 datos en total, asignamos el 30%  de ellos al conjunto de validación. Esperando que esta decisión nos permita tener una muestra representativa para evaluar el rendimiento final de nuestros modelos.
Además, al analizar la distribución de las clases en nuestros datos, observamos que el 68.6% pertenece a la clase 0 (mal pronostico) y el 31.4\% a la clase 1 (buen pronostico). Para mantener esta proporción en nuestros conjuntos de desarrollo y validación, aseguramos que la misma proporción de cada clase esté presente en ambos conjuntos.

Por ultimo, para evitar cualquier tipo de sesgo, antes de realizar la asignación, aplicamos un random shuffle al conjunto de datos, es decir, mezclamos aleatoriamente los datos antes de dividirlos. Esto garantiza que no haya ningún orden preestablecido en nuestros datos que pueda influir en la separación.

## Construcción de modelos: Árboles de Decisión

### Primer modelo
Entrenamos un árbol de decisión con una altura máxima de 3 y los demás hiperparámetros en sus valores por defecto. Luego, estimamos su rendimiento utilizando validación cruzada  estratificada con $K$ = 5. Esta técnica nos permite respetar las proporciones mencionadas en el punto anterior.

Calculamos métricas como la precisión, el área bajo la curva de precisión-recall (AUPRC) y el área bajo la curva ROC (AUCROC) para cada fold por separado, así como su promedio. Además, calculamos el score global para los folds de validación. Esta métrica global consiste en tomar el vector de resultados de cada fold y concatenarlo en un único vector de predicciones. Luego, evaluamos el desempeño del modelo utilizando este vector combinado, lo que nos permite obtener una medida agregada del rendimiento del modelo en todos los pliegues de validación.

**Rendimiento del árbol de decisión** 

| Fold      | Train Accuracy | Test Accuracy | Train ROC AUC | Test ROC AUC | Train Avg Precision | Test Avg Precision |
|-----------|----------------|---------------|---------------|--------------|---------------------|--------------------|
| 1         | 0.832          | 0.671         | 0.862         | 0.613        | 0.745               | 0.449              |
| 2         | 0.818          | 0.657         | 0.787         | 0.657        | 0.663               | 0.460              |
| 3         | 0.846          | 0.657         | 0.872         | 0.624        | 0.718               | 0.416              |
| 4         | 0.846          | 0.629         | 0.860         | 0.563        | 0.772               | 0.350              |
| 5         | 0.854          | 0.700         | 0.816         | 0.519        | 0.692               | 0.372              |
| Promedio  | 0.839          | 0.663         | 0.840         | 0.595        | 0.718               | 0.410              |
| Global    | -              | 0.663         | -             | 0.584        | -                   | 0.367              |

La baja varianza entre las mediciones de nuestros folds sugiere que no existe un 'fold' predominante que desvíe los datos. Esta consistencia en las mediciones brinda una mayor confianza en la calidad de la separación de los datos realizada durante el proceso de validación cruzada. Comparando los Cuadros, observamos que obtuvimos resultados satifactorios en la etapa de entrenamiento pero no asi en la de testeo, con cualquiera de las 3 métricas calculadas. 


### Explorando algunos hiperparámetros

Para cada combinación de hiperparámetros validamos sus resultados como en el punto anterior.
** Rendimiento del árbol de decisión con diferentes combinaciones de hiperparámetros **
| Combinación de hiperparámetros | Criterio de corte | Train Accuracy | Test Accuracy |
|--------------------------------|-------------------|----------------|---------------|
| 3                              | gini              | 0.839          | 0.663         |
| 5                              | gini              | 0.939          | 0.654         |
| Altura infinita                | gini              | 1.000          | 0.637         |
| 3                              | entropy           | 0.800          | 0.646         |
| 5                              | entropy           | 0.912          | 0.643         |
| Altura infinita                | entropy           | 1.000          | 0.626         |



Al explorar modelos con diferentes alturas $(h)$ de árbol, en el Cuadro se evidencia una tendencia interesante pero esperada, utilizando cualquiera de los dos criterios: el aumento de $h$ parece mejorar el rendimiento en el conjunto de entrenamiento, pero al mismo tiempo deteriora el rendimiento en el conjunto de evaluación. Este fenómeno indica que, al permitir que los árboles crezcan más profundamente, se produce un sobreajuste a los datos de entrenamiento, lo que resulta en una mala generalización a nuevos datos. Al comparar los criterios de separación '$gini$' y '$entropy$', notamos una muy ligera ventaja en el rendimiento del criterio '$gini$' en el conjunto de evaluación.


## Comparación de algoritmos

En esta instancia probamos distintas combinaciones de hiperparámetros para cada modelo mediante RandomizedSearchCV usando como métrica de performance AUCROC resultante de stratified 5-fold cross validation. 
Este método se diferencia de GridSearch ya que no especificamos una grilla de hiperparámetros si no que para cada hiperparámetro asignamos una distribución de probabilidad a muestrear.

### Random Search vs Grid Search
La ventaja de RandomizedSearchCV es que tenemos mayor flexibilidad a la hora de explorar el espacio de hiperparámetros, por lo que existe la posibilidad de que se encuentren máximos locales que podrían no haber sido un valor de la grilla. A su vez, si el espacio de hiperparámetros no es muy grande, un número menor de combinaciones con RandomizedSearchCV puede hacer una exploración razonablemente buena del espacio y darnos resultados similares. Por lo que el costo computacional es menor. 

Inicialmente podría parecer que expandir el espacio de hiperparámetros lo más posible es mejor, ya que podremos explorar más valores. Pero mientras más lo expandamos, mayor es el tamaño de muestra que debemos samplear para que sea representativa, es decir, tendriamos que aumentar considerablemente el número de combinaciones a probar, lo cual aumenta el costo computacional. 

Es por esto que resulta necesario hacer un análisis de los hiperparámetros para restringir las distribuciones solo a valores lógicos que podrían tomar y también seleccionar los hiperparámetros más importantes. Los que no se especifican toman un valor por default.
En todos los casos se probaron 50 puntos distintos.

### Árboles de decisión
A continuación se presentan las distribuciones que se utilizaron para muestrear cada hiperparámetro. Las listas de elementos son interpretadas como distribuciones uniformes discretas
- criterion $\sim$ [Gini, Entropy, Log Loss]: El criterio utilizado para medir la calidad de la partición realizada.
- max depth $\sim U(1,40)$ (Discreta): Máxima profundidad que puede tomar el árbol.
- min samples split $\sim U(2,20)$ (Discreta): La mínima cantidad de datos necesaria para partir un nodo.
- min samples leaf $\sim U(1,20)$ (Discreta): La mínima cantidad de datos necesaria para que un nodo sea hoja. En cada partición, ambos hijos tienen al menos esta cantidad de datos.
- max features $\sim U(1,50)$ (Discreta): La máxima cantidad de features considerados en cada partición.
- max leaf nodes $\sim U(2,|X train|)$ (Discreta): Máxima cantidad de hojas que puede tener el árbol.

Los valores del mejor modelo obtenido son los siguientes:

| Hiperparámetro    | Valor Óptimo |
|-------------------|--------------|
| Criterio          | Log Loss     |
| Max Depth         | 10           |
| Min Samples Leaf  | 14           |
| Max Features      | 20           |
| Max Leaf Nodes    | 269          |

**Tabla:** Mejor modelo TREE, con AUCROC = 0.638

La profundidad limitada del árbol y la cantidad mínima de instancias en una hoja ayudan a evitar el sobreajuste.
### KNN
A continuación se presentan las distribuciones sobre las que se muestreó cada hiperparámetro.

- metric $\sim$ [Euclidean, Manhattan, Chebyshev, Minkowski]: La métrica utilizada para calcular las distancias a los puntos.
- n neighbors $\sim U(1,30)$ (Discreta): La cantidad de vecinos considerada al clasificar un nuevo punto.
- p $\sim U(1,5)$ (Discreta): Parámetro de "poder" de la métrica de Minkowski si la usamos.
- weights $\sim$ [Uniforme, Distancia]: Define si todos los vecinos tienen el mismo peso (Uniforme) o se les asignan distintos pesos según su distancia (Distancia).

Los valores del mejor modelo obtenido son los siguientes:

| Hiperparámetro | Valor Óptimo |
|----------------|--------------|
| N Neighbors    | 26           |
| Weights        | distance     |
| P              | 4            |
| Metric         | euclidean    |

El uso de la distancia ponderada 'distance' y un número moderado de vecinos ayuda a reducir el efecto de los puntos lejanos, mejorando así la capacidad de generalización del modelo.

### SVM
Distribuciones sobre las cuales se optimizaron hiperparámetros:

- C $\sim U(0,10)$ (Continua): Parámetro de regularización. La fuerza de la regularización es inversamente proporcional a C.
- cache size: $\sim U(1,1000)$: El tamaño del cache del kernel en MB
- coef0 $\sim U(0,1)$ (Continua): Término independiente en la función kernel. Solo es significante con los kernels Polinomial y Sigmoide
- degree $\sim U(1,6)$ (Discreta): Grado del polinomio de la función kernel. Solo es significante con el kernel Polinomial
- kernel $\sim$ [Lineal, Polinomial, RBF, Sigmoide]
- tol $\sim U(0,0.1)$ (Continua): Tolerancia para el criterio de parada.

Los valores del mejor modelo obtenido son los siguientes:

| Hiperparámetro | Valor Óptimo |
|----------------|--------------|
| C              | 3.061        |
| Kernel         | poly         |
| Degree         | 2            |
| Tol            | 0.045        |
| Cache Size     | 209          |
| coef0          | 0.843        |

La elección del kernel polinomial de grado 2 y un coeficiente de regularización C óptimo de 3.061 sugiere que el modelo se adapta bien a la no linealidad de los datos.

### LDA
Distribuciones sobre las cuales se optimizaron hiperparámetros:

- Priors $\sim U(0,1)^2$ (Continua): Las probabilidades a priori de cada clase
- tol $\sim U(0,0.1)$ (Continua): El threshold para un valor para ser considerado significante.

Los valores del mejor modelo obtenido son los siguientes:

| Hiperparámetro | Valor Óptimo |
|----------------|--------------|
| Priors         | [0.087, 0.913] |
| Tol            | 0.375        |

### Naïve Bayes
Distribuciones sobre las cuales se optimizaron hiperparámetros:

- Priors $\sim U(0,1)^2$ (Continua): Las probabilidades a priori de cada clase
- var smoothing $\sim U(0,0.1)$ (Continua): Porción de la variación más grande de todas las características que se agrega a las variaciones de cada característica para la estabilidad del cálculo.

Los valores del mejor modelo obtenido son los siguientes:

| Hiperparámetro | Valor Óptimo |
|----------------|--------------|
| Priors         | [0.087, 0.913] |
| Var Smoothing  | 0.0187       |


## 4. Diagnóstico Sesgo-Varianza

### Curvas de Aprendizaje

De acuerdo a las mejores configuraciones obtenidas en el punto anterior, graficamos la curva de aprendizaje de 3 modelos.

![Curvas de aprendizaje para Árboles de Decisión, LDA y SVM](/imagenes/output.png)

Representan la relación entre el rendimiento del modelo y la cantidad de datos utilizados para entrenarlo. Al visualizar una curva de aprendizaje, podemos observar cómo mejora o se estabiliza el rendimiento del modelo a medida que se incrementa la cantidad de datos de entrenamiento.

En el caso del LDA, parece haber cierto margen aún para mejorar el modelo agregando datos de entrenamiento, ya que la curva de aprendizaje de test score sigue teniendo una tendencia creciente durante todo el gráfico. En cambio, las curvas de Árboles de Decisión y de SVM indican cierta saturación, por lo que es poco probable que se consigan mejoras significativas con más datos de entrenamiento (lo cual valida nuestra decisión de haber seleccionado únicamente el 70 por ciento para el entrenamiento).

### Interpretación de sesgo, varianza, sub y sobre ajuste

Si la curva de entrenamiento muestra un alto rendimiento pero la curva de validación muestra una brecha significativa con respecto a la de entrenamiento, puede estar indicando alguno de los siguientes casos:

- Sesgo en nuestro modelo: por ejemplo, este es el caso del gráfico de SVM, en el que podemos ver que las curvas de train y test parecen haberse estabilizado y la brecha entre una y otra se mantiene constante aunque se agreguen nuevos datos. Por ejemplo, al observar el gráfico de LDA y SVM vemos que estamos en casos con algo de sesgo, pero si comparamos los AUC scores que se obtuvieron podemos decir que LDA está subajustando respecto a SVM.
- Alta varianza en nuestro modelo: esto ocurre cuando la curva de test es 'ruidosa' o poco suave, lo que nos indica que a medida que agrandamos la muestra las predicciones fluctúan mucho. Este es el caso del gráfico del árbol de decisión y, en menor medida, en LDA. También podemos observar el sobreajuste, ya que el Árbol obtuvo una performance respetable en la etapa de entrenamiento pero cuando tuvo que hacer predicciones nuevas - al contrario de los otros dos - no fue capaz de generalizar, indicando que probablemente aprendió el set de entrenamiento y no así los patrones más generales.

## 4. Diagnóstico Sesgo-Varianza

### Curvas de Complejidad

A continuación se presentan las curvas de complejidad de Árboles de decisión y SVM. En ambos casos, se mantuvieron todos los hiperparámetros de los mejores modelos hallados (en cuadros 4 y 6 respectivamente), variando uno solo de ellos en cada caso: la profundidad máxima para árboles y el parámetro C para SVM.

![Curvas de complejidad para Árboles de Decisión y SVM. Los valores que minimizan el error en el set de evaluación son max depth = 4 para Árboles y c = 3.051 para SVM.](/imagenes/complejidad-arbol-svm.png)

Estos son útiles para estudiar cómo afecta un hiperparámetro a la performance del modelo. En ambos casos (para C y la profundidad máxima) observamos un comportamiento similar en cuanto a su punto óptimo: para valores muy pequeños del parámetro, aumentarlo es beneficioso porque aún tenemos mucho error en el conjunto de entrenamiento. Sin embargo, también se observa que existe un punto en el que el decrecimiento del error en las curvas se detiene. En el caso de los Árboles, parece rápidamente alcanzar un comportamiento constante o asintótico, mientras que en el caso de SVM parece primero tener un leve crecimiento luego del mínimo, antes de alcanzar el régimen asintótico.

En ambos casos, a partir de un momento aumentar el valor del parámetro no genera mejoras significativas en el resultado del conjunto de evaluación.

Esto está intrínsecamente relacionado con el subajuste o sobreajuste, ya que a medida que el modelo se vuelve más complejo (es decir, el hiperparámetro se vuelve más expresivo), puede aprender más detalles de los datos de entrenamiento, lo que resulta en un mejor rendimiento en este conjunto de datos. Sin embargo, este aumento en la performance no se traslada necesariamente a la etapa de prueba, lo que resulta en una generalización deficiente en datos no vistos.

### Random Forest

Construimos un modelo RandomForest con 200 árboles en base a nuestro conjunto de entrenamiento.

En la Figura \ref{complejidad_rf} presenta la curva de complejidad para el hiperparámetro $max\_features$, manteniendo los demás hiperparámetros en el valor default de scikit-learn.

![Curvas de complejidad para max_features en Random Forest con 200 árboles.](/imagenes/complejidad-rf.png)

Mientras que en el set de entrenamiento el error es siempre nulo (es decir, el valor de AUC da siempre 1), el error en test comienza con tendencia decreciente.

El valor predeterminado de este hiperparámetro es $\sqrt{200} \approx 14$. En el gráfico se observa que si bien el mínimo absoluto se alcanza en $max\_features = 7$, este parece ser una excepcionalidad, ya que la tendencia sigue decreciente hasta aproximadamente 16. Es decir, el valor predeterminado parece razonable.

A partir de ahí, el error parece mantenerse aproximadamente constante o con una leve tendencia creciente. Es decir, ya no se achica la diferencia entre la performance en entrenamiento (que es siempre 0) y en evaluación, por lo que estamos ante un posible caso de sobreajuste.

Limitar el subconjunto de características en cada nodo permite diversificar los árboles, alcanzando mayor robustez y así también un mejor poder de generalización del bosque.

También graficamos la curva de aprendizaje con $n=200$ árboles y $max\_features$ en su valor predeterminado.

![Curva de aprendizaje para Random Forest con 200 árboles](/imagenes/learning-rf.png)

En esta figura podemos observar que nuevamente el training test alcanza un roc auc de 1.00 en todo momento. En el test de evaluación, en cambio, se aprecia una mejora de la performance al aumentar el tamaño del subconjunto considerado. Aproximadamente en 110 parece alcanzar un máximo, y a partir de ahí parece quedar constante: por más datos de entrenamiento que se agreguen, el score no parece mejorar.

Si bien la brecha entre ambas curvas se mantiene constante, lo que indica que el sesgo no disminuye, con más datos la curva sí parece suavizarse y estabilizarse, lo que implica una posible disminución de la varianza.

## 5. Evaluación del Desempeño del Modelo

El modelo seleccionado es el SVM, según los hiperparámetros encontrados en el Cuadro 6 y entrenado con la totalidad del set de entrenamiento.

### AUC-ROC sobre el conjunto de evaluación

El modelo SVM entrenado obtuvo un AUC-ROC de 0.870 sobre el conjunto de datos de validación (es decir, el 30% del dataset original que fue separado en un comienzo). Esto indica que es esperable que el modelo tenga una buena capacidad para discriminar entre las clases positivas y negativas.

### Comentarios Adicionales

- La alta AUC-ROC obtenida tanto en el conjunto de validación como en el análisis de bootstrap sugiere que el modelo SVM es capaz de hacer predicciones precisas y consistentes.

## 6. Conclusiones

En este trabajo de investigación se exploró la aplicación del aprendizaje automático para analizar datos de lesiones pretumorales, para identificar biomarcadores para el diagnóstico temprano y la predicción del pronóstico del cáncer.

### Investigación con Machine Learning

- Se empleó un modelo de aprendizaje automático basado en SVM con optimización de hiperparámetros mediante Random Search para clasificar pacientes con buen y mal pronóstico a partir de datos de la expresión génica. El modelo entrenado obtuvo un AUC-ROC de 0.870 en el conjunto de validación, indicando una buena capacidad para discriminar entre las clases.
- Se realizó un análisis de bootstrap para evaluar la estabilidad del modelo, sugiriendo confiabilidad y generalizabilidad de sus resultados.
- Se compararon diferentes algoritmos de machine learning (árboles de decisión, KNN, LDA, NaiveBayes, Random Forest) para identificar el más adecuado para este problema específico. SVM se destacó como el modelo con mejor rendimiento.
- Se analizaron las curvas de aprendizaje y las curvas de complejidad para comprender el comportamiento de los modelos y optimizar su rendimiento. Se observó que LDA podría beneficiarse de más datos de entrenamiento, mientras que SVM y los árboles de decisión parecen estar cerca de la saturación. Las curvas de complejidad indicaron la existencia de un punto óptimo para la complejidad del modelo, donde se maximiza el rendimiento sin comprometer la generalización.

### Implicancias para el Diagnóstico y Pronóstico del Cáncer

- La identificación de biomarcadores a partir de datos de expresión génica pre-tumoral podría beneficiar la precisión y personalización de las herramientas de diagnóstico para el cáncer, basadas en el perfil molecular de cada paciente. Al promover la predicción temprana se pueden mejorar las tasas de supervivencia.

### Limitaciones y Futuras Investigaciones

- El estudio se basa en un conjunto de datos relativamente pequeño, por lo que se requieren estudios de validación con cantidades más amplias para confirmar los hallazgos.
- Se podrían explorar técnicas de machine learning más avanzadas, como redes neuronales profundas, para mejorar aún más el rendimiento de los modelos.
