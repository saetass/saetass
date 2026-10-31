# Cómo reproducir saetass en GPU: PINN, formulación débil, operadores neuronales y solvers

Investigación del 31 de octubre de 2026, rama `PINN`. Código en este directorio y derivaciones en [NOTES.md](NOTES.md). La [tercera ronda](#tercera-ronda-el-solver-en-saetasspinn) corrige varias conclusiones de las dos primeras.

## Conclusión

**La formulación débil funciona y arregla lo que fallaba, pero la mejor opción para saetass no es una red: es la propia formulación débil resuelta directamente en GPU.** La ecuación es lineal, y eso permite obtener `ψ(t)` con unas pocas resoluciones tridiagonales, sin pasos de tiempo. Las redes solo merecen la pena en un nicho.

| Pregunta | Respuesta |
|---|---|
| ¿Arregla la forma débil las discontinuidades? | Sí, si se usa la norma dual con test exponencial. La VPINN clásica y el residuo fuerte fallan. |
| ¿Sirve un operador neuronal para discontinuidades fuertes? | Sí, con envolvente numérica: 5,4 % con contraste ×10⁶, frente a 63 % sin ella. |
| ¿Se puede entrenar sin datos? | En un solo problema, sí (0,13-1,2 %). En un operador general, no con estas pruebas. |
| ¿Gana alguna red al solver? | En estático, no. En perfiles temporales, empate al 1 %. En muestreo masivo de una familia fija, sí, unas 12 veces. |
| ¿Cubre el solver a saetass? | En casi todo lo probado. En `electrons_2d` coincide el número de partículas por debajo de 400 GeV (0,02 %), no la forma del frente sin resolver. |

## Comparación a 256 nodos, GPU libre

| Método | µs por perfil | Error L2 mediano | Perfiles arbitrarios | Entrenamiento |
|---|---|---|---|---|
| Contorno, 2 resoluciones complejas | 3,7 | 0,8 % | sí | ninguno |
| Contorno, 3 | 5,4 | 0,06 % | sí | ninguno |
| Contorno, 4 | 7,2 | 0,004 % | sí | ninguno |
| Contorno, 12 | 21 | 1e-12 frente a N=32 | sí | ninguno |
| FNO con envolvente | ~37 | 0,9-1,1 % | sí | 10-20 min |
| FNO, burbujas incluidas en el entrenamiento | ~37 | 0,35 %, p95 0,8 % | sí | 57 min |

Son perfiles con la malla resuelta. Los errores del contorno se miden frente a sí mismo con N=32, y los del FNO frente al contorno.

## Por qué fallaba y qué lo arregla

En forma fuerte hay que imponer a mano los saltos, la fuente delta es singular y la red debe dar segundas derivadas en capas de ancho `D/v`. En forma débil,

    ∫ x² ∂ψ/∂t φ dx − ∫ x² (v ψ − D ∂ψ/∂x) ∂φ/∂x dx = φ(1)

la delta es `φ(1)`, los saltos son condiciones naturales y solo hay una derivada. Pero eso no basta: hay que medir el residuo en su **norma dual**, `Rᵀ G⁻¹ R`, porque es la que acota el error (RVPINN). La mejor `G` es `I − tA` con el operador de Scharfetter–Gummel, que es un Petrov-Galerkin con test exponencial.

Misma burbuja y misma red, sin datos, 12 000 pasos. Error L2 a `t = 100`:

| Pérdida | `Pe = 1` | `Pe = 22` | `Pe = 100` |
|---|---|---|---|
| Residuo fuerte | 100 % | 100 % | 100 % |
| VPINN (`G = I`) | 56 % | 100 % | 100 % |
| Dual de energía | 100 % | 0,55 % | 17 % |
| **Dual Scharfetter–Gummel** | **0,34 %** | **0,13 %** | **0,17 %** |

El residuo fuerte y la VPINN se quedan en la solución trivial `ψ ≈ 0`. La norma de energía queda sin juzgar del todo: a `Pe = 1` no arranca, posiblemente por escala, y el reintento con el logaritmo de la pérdida no sirve de árbitro, porque colapsa en las cuatro pérdidas, incluida `sg`. La pérdida `sg` tarda unos 40 minutos y sigue mal en las colas (`|Δ ln ψ|` ≈ 5 a `Pe` alto). Es un resultado de método, no una alternativa práctica: el solver hace lo mismo en milisegundos.

## Operadores neuronales

La idea que cambia los resultados es dar a la red una **envolvente numérica**: para cualquier perfil, `ψ_ss = −A⁻¹F` o un paso implícito `ψ_BE = (I − tA)⁻¹ t F` cuestan una resolución. La red aprende el cociente.

FNO con 256 nodos, 20 000 pasos, entrenado sin burbujas. Las filas con rango son tres semillas; las demás, una:

| Envolvente y pérdida | Aleatorios | Burbujas no vistas | p95 burbujas |
|---|---|---|---|
| Ninguna, datos | 2,8 % | 10,9 % | 35 % |
| `ψ_ss`, datos | 1,0-1,3 % | 1,4-2,4 % | 38-48 % |
| `ψ_ss`, datos + débil | 0,8-0,9 % | 0,9-2,0 % | 17-29 % |
| `ψ_BE`, datos | 0,9-1,0 % | 2,1-2,5 % | **5-7 %** |
| `ψ_BE`, datos + débil | 1,4-1,5 % | 2,8-3,6 % | 11-16 % |
| `ψ_BE`, solo física | 4,8 % | 7,4 % | 22 % |
| `ψ_ss`, solo física | 5,0 % | 3,9 % | 451 % |

La envolvente sola da 12 % en aleatorios, y 2 % (`ψ_ss`) o 15 % (`ψ_BE`) en burbujas. Por tanto, **`ψ_ss` solo con física es peor en burbujas que no usar red**. La pérdida débil solo ayuda a `ψ_ss`; a `ψ_BE` le perjudica.

Sobre discontinuidades, sin reentrenar y entrenado hasta ×30, L2 mediano y p95:

| Contraste de `D` | Sin envolvente | `ψ_BE` |
|---|---|---|
| ×10² | 11 % / 32 % | 2,1 % / 5,1 % |
| ×10⁴ | 31 % / 52 % | 2,9 % / 7,4 % |
| ×10⁶ | 63 % / 82 % | 5,4 % / 13,9 % |

Otros casos, con envolvente:
- **Parámetros temporales:** 1,0 % (burbujas 2,0 %) en 0,22 ms. El solver por pasos a igual coste da 0,65 %. Empate en torno al 1 %.
- **`(r, p)` con pérdidas:** 0,77 % (burbujas 1,0 %, espectro 0,4-0,6 %) en 1,65 ms, frente a 4,3 ms del contorno, medidos con GPU compartida. Sin envolvente: 3,5 % y 8,6 %.
- **DeepONet:** 9,8 % y 45 % en burbujas, peor que el FNO, como predice la cota de Lanthaler et al.

## El solver contra saetass

| Caso | Solver | saetass |
|---|---|---|
| Aceleración en choque, pendiente (teoría −2,0563) | **−2,0565**, 1,5 s en CPU | −2,0775, 42 s con 8 hilos |
| Amplitud del choque (estimación a mano ~0,885) | 0,8978, ya convergida | 0,813 y 0,821, aún subiendo |
| `full_2d` con pérdidas, `N/Ninj` a `p = 3e5` | 0,4268 | 0,4851 a `dt` del benchmark, 0,4268 a `dt/100` |
| `protons_2d`, pérdidas reales | 3,5 s | converge al solver, L2 0,0 % a `dt/64` |
| `electrons_2d`, Bohm, `Pe` de celda 1e8 | **no coincide** | L2 8,5 %, p95 100 %, sin mejorar con `dt` |

Búsqueda de parámetros con datos de saetass: solver y red dan el mismo posterior (`L_wind` 38,303, `η_B` −1,013 y −1,015, χ² 36,7). Solver a 0,10 ms por perfil, red a 8,5 µs: la red amortiza sus 3,6 h de entrenamiento a partir de unos 130 millones de perfiles.

El benchmark `full_2d` tiene un 9 % de error por paso de tiempo a su `dt`, y da `N/Ninj = 1,015` con escape, algo imposible. Es el mecanismo de #74.

## Segunda ronda: qué sirve para publicar

### Búsqueda de parámetros de grueso a fino

Mismos datos simulados con saetass. El modelo genérico acepta cualquier `D(r)`, `v(r)` y fuente, sin familia ni parametrización.

| Etapa burda | Etapa final | ESS | Corridas finales | Resultado |
|---|---|---|---|---|
| 128 nodos + discrepancia lineal | solver 1024 nodos | 1 de 1024 | 1792 puntos | **colapsa** |
| 256 nodos + discrepancia lineal | solver 1024 nodos | 12 de 1024 | 1792 puntos | sesgo < 0,5σ, χ² 37,4 |
| 512 nodos + discrepancia lineal | solver 1024 nodos | 681 de 1024 | 1792 puntos | sesgo < 0,8σ, χ² 48,3 |
| **Solver 1024 nodos** | **saetass real, 1200 celdas** | **214 de 256** | **1024 corridas, 281 s en 16 núcleos** | **posterior exacto de saetass**, ≤ 0,7σ |

Hacerlo directamente costaría unas 980 horas-núcleo. Son 1024 corridas en vez de 820 000.

Como modelo burdo, el FNO genérico no compensa: en una burbuja física fuera de su rango da 11,7 % (5,2 % sin sesgo medio), frente a 4,5 % de 256 nodos, sin entrenar y 12 veces más barato. Reentrenarlo para ese rango lo mejoraría, pero dejaría de ser independiente de las funciones.

### Inversión de una función sin suponer su forma

El gradiente a través del contorno coincide con diferencias finitas en todas las cifras mostradas. Con eso se recupera `D(r)` como 160 nodos libres, con 4 energías, 40 radios, ruido del 5 % y datos de saetass, en 11 minutos. Ver [la figura](fig_function_inference.png).

- χ²/N = 0,99 y error mediano del 2-3 % en cada zona, con un salto de 260 veces.
- El salto de `R_b` cae unos 5 pc desplazado, y junto a él el error llega a un factor 86. Es comparable a la separación de los datos allí.
- Las bandas no son fiables en el viento (cobertura 28 % para un 68 % nominal). En burbuja y medio, 91 % y 95 %.

### Forma débil sin malla global

Sin datos, centros aleatorios, nube de puntos redibujada en cada paso y solo las posiciones de los saltos como información. Error L2 de `t = 1` a `t = 300`:

| Variante | `Pe = 1` | `Pe = 22` | `Pe = 100` |
|---|---|---|---|
| Con malla, dual Scharfetter–Gummel (referencia) | 0,34-0,78 % | 0,13-1,1 % | 0,17-1,2 % |
| **Sin malla, dual con funciones radiales** | **4-61 %** | **6,5-17 %** | **3,7-7 %** |
| Sin malla, norma diagonal y test exponencial | 10-28 % | sin terminar, 100 % a media corrida | no ejecutado |
| Sin malla, norma diagonal | no ejecutado | 49-69 % | no ejecutado |

Funciona a `Pe` alto, y falla a `Pe = 1` en tiempos tardíos, quizá por falta de funciones de test anchas, sin comprobar.

### Valoración

| Ventaja posible | Fuerza | Por qué |
|---|---|---|
| Solver diferenciable por lotes en GPU, sin pasos de tiempo | **Alta** | No encontré un solver diferenciable de transporte de rayos cósmicos |
| Inversión de `D(r)` sin suponer forma | **Alta** | Imposible con redes por familia, y con saetass inabordable |
| Posterior exacto de saetass con ~1000 corridas | **Alta** | Unas 800 veces menos corridas |
| Mesh-free | **Baja hoy** | 5-50 veces peor que con malla, y un solo problema resuelto en 50 minutos |
| Operador neuronal independiente de funciones | Baja como burdo | Pierde contra baja resolución |

La búsqueda bibliográfica fue limitada: "no encontré" no significa que no exista. Antes de reclamar novedad hace falta una revisión completa. Además, todo se probó con una sola realización simulada de una burbuja, sin datos reales.

## Tercera ronda: el solver en `saetass.pinn`

El solver ya está en [radial_solver.py](../src/saetass/pinn/radial_solver.py) y la inversión en [inversion.py](../src/saetass/pinn/inversion.py), con 45 tests. Respecto al prototipo tiene tres cambios de método.

1. **Variable `y = ψ/ψ_ss`.** El estacionario exacto de la malla se calcula en logaritmos con `logcumsumexp`, sin desbordar, y la comprobación `0 ≤ y ≤ 1` se hace nodo a nodo.
2. **Flujos ajustados en el dominio de Laplace.** En cada `s` del contorno, cada celda usa las soluciones exponenciales de `Dψ'' − vψ' − sψ = 0`. A `s = 0` se reduce a Scharfetter–Gummel, pero también ajusta el transitorio, así que un frente no sufre difusión numérica de primer orden.
3. **Malla propia por burbuja**, con nodos en el choque y en `R_b`, e integrales de flujo exactas. Un radio arbitrario se evalúa con el `ψ_ss` cerrado por la interpolación de `ln y`, sin malla de salida.

### Precisión, en el conjunto de prueba de #73

| Modelo | L2 mediana | p95 | Peor | Coste |
|---|---|---|---|---|
| Red (media de dos) | 0,08 % | 0,41 % | 2,2 % | 8,5 µs, 3,6 h de entrenamiento |
| Solver, 128 nodos | 0,076 % | 0,97 % | 4,0 % | 0,14 ms |
| Solver, 512 nodos | 0,0047 % | 0,062 % | 0,25 % | 0,14 ms |
| Solver, 1024 nodos | 0,0012 % | 0,014 % | 0,057 % | 0,17 ms |

Sin el ajuste en Laplace, a 512 nodos: p95 0,20 % y peor 1,3 %, en 256 burbujas aleatorias frente a 0,048 % y 0,14 % con él. `|Δ ln ψ|` por encima de 1e-6 del máximo, frente al mismo solver a 4096 nodos: con 1024 nodos, mediana 0,019 %, p95 1,1 %, peor 3,2 %. Eso cumple los dos criterios de #73, que la red incumple, pero la comparación no es independiente.

Contra la referencia del conjunto de prueba, el peor `|Δ ln ψ|` se queda en 24-29 %. **Ese error es de la referencia**: en esa burbuja la diferencia con el solver es 325 %, 93 %, 25 % y 6,5 % a resoluciones 1, 2, 4 y 8. Por tanto el 13,9 % de la red tampoco mide bien su error real.

### saetass converge al solver en el viento

A 100 GeV, 3 pc, `|Δ ln ψ|` frente a saetass: 33 % con 1200 celdas, 8 % con 2400, 1,8 % con 4800 y 0,15 % con 9600. A 10 TeV con 2400 celdas, 8,3 % con 12000 pasos, 0,27 % con 84000 y 0,12 % con 300000.

### Búsqueda de parámetros, mismos datos, 512 caminantes y 400 pasos

| Nodos | Búsqueda | ESS con 1024 corridas de saetass |
|---|---|---|
| 128 | 22 s | 134 de 256 |
| 256 | 24 s | 152 de 256 |
| 512 | 32 s | 152 de 256 |
| 1024 | 86 s | 152 de 256 |

**El modelo burdo de 128 nodos ya no colapsa**: el prototipo daba ESS 1 de 1024 a esa resolución. De 256 a 1024 nodos la cadena es idéntica, porque ninguna aceptación cambia. La corrección con saetass tarda 282 s en 16 núcleos y deja desvíos ≤ 0,65σ.

El ESS no sube de 152, frente a 214 del prototipo. Mi hipótesis, sin comprobar, es que el límite es el error de saetass a 1200 celdas, que la corrección lleva a la posterior: un saetass más fino podría subirlo y además cambiaría la "posterior de saetass".

### Inversión de `D(r)`

160 valores libres, 32 realizaciones de ruido, datos de saetass a 4800 celdas, ruido del 3 %.

| Datos | χ²/N | Error mediano viento / burbuja / medio | Cobertura banda 68 % | Salto en `R_b` (86,6 pc) | Tiempo |
|---|---|---|---|---|---|
| 40 radios | 0,92 | 0,22 % / 1,0 % / 0,69 % | 94 / 93 / 94 % | 93,3 pc | 7,5 min |
| 160 radios | 1,00 | 0,40 % / 0,99 % / 1,3 % | 75 / 89 / 95 % | **86,9 pc** | 14 min |

Queda un artefacto nuevo a la vista en la figura: justo tras el choque, entre 14 y 20 pc, `D` oscila hasta un factor 2,2 y la banda no lo cubre. El resumen por zonas empieza en 15,7 pc, así que el error mediano no lo refleja.

Con esto se corrigen dos fallos de la segunda ronda:

- **Cobertura del 28 % en el viento.** Lo más probable es que viniera del error de saetass a 1200 celdas. Con 4800 celdas sube a 75-94 %. Las bandas siguen sin estar calibradas: salen anchas, no al 68 %.
- **Salto desplazado.** Era la separación de los datos, 10 pc, y no el método: con cuatro veces más radios cae a un nodo de distancia, separación de 1,9 pc. Junto al salto, dentro de esa celda, el error sigue siendo de hasta un factor 157 de un salto real de 265.

### `electrons_2d`

No era el agujero que describí. Con el solver por barrido en `p`, el número de partículas coincide con saetass en 0,02 % hasta 400 GeV. El L2 de 8,7 % viene de la forma del frente, a unas 7 celdas del choque, sin resolver en ninguno de los dos. A 2,5 TeV hay un 8 % de diferencia en número, y sobre 600 TeV solo coinciden en que `ψ` es despreciable, 1e-20 y menor.

**El ajuste en Laplace no sirve aquí.** `ψ̂ ∝ e^{−sτ}` crece en la parte izquierda del contorno si el retardo supera a `t`, y desborda. El contorno no puede representar un frente de advección casi pura que aún no ha llegado. El ajuste tampoco está unido a las pérdidas: `saetass.pinn` cubre solo el caso sin pérdidas.

### Pendiente

- Pérdidas, parámetros temporales y Bohm en la burbuja (`δ = 1`) no están en el módulo.
- Una sola burbuja simulada, sin datos reales, y revisión bibliográfica limitada, también para el ajuste en Laplace, que tiene antecedentes en elementos analíticos con transformada de Laplace.
- La documentación no se ha compilado: Sphinx no está instalado en este entorno.

## Recomendación

1. **Núcleo:** un operador lineal general en `(r, p)` en GPU por lotes, con esta forma débil. Tres modos: contorno para coeficientes estáticos, factorización o GMRES cuando `ṗ` cambia de signo (47 iteraciones, sin depender de la malla) y pasos implícitos para parámetros temporales.
2. **Control de errores:** comprobar `0 ≤ ψ ≤ ψ_ss` y `Pe` de celda, y pasar los casos marcados a pasos implícitos.
3. **Red, opcional y al final:** solo para campañas de cientos de millones de evaluaciones de una familia fija, con envolvente `ψ_BE` y datos del solver.
4. **Descartado:** PINN puro, VPINN clásica y DeepONet.

## Límites

- **Advección sin resolver.** Con `Pe` de celda < 1 hubo 0 fallos en 7512 perfiles. Con 1-10, 1,5 %; con 10-100, 8 %. `electrons_2d` no coincide con saetass. La causa probable, que no he comprobado, es que Scharfetter–Gummel degenera en upwind de primer orden. Es el agujero más importante.
- **Detector.** Con N=24 atrapó 14 de 14 fallos; con N=16, 37 de 40. Solo vale para fuente constante y coeficientes estáticos.
- **Colas.** A `Pe ≈ 290`, error absoluto ~1e-5 del máximo. Valores negativos posibles con upwind de segundo orden (hasta 1,5 % cerca de la inyección).
- **Sin malla.** El solver da nodos: interpolar a radios arbitrarios deja L2 de 1,6e-5, pero el error puntual en colas llega al 100 % con 1401 nodos y baja a 4 % con 5601.
- **No probado:** adiabáticas dependientes del tiempo en lotes GPU, GMRES en GPU, contorno por lotes en mallas reales de saetass, wPINN con red de test, IONet, TFPONet, transformadores, optimizadores de segundo orden. Las redes tuvieron presupuesto limitado.
- **Saetass como árbitro.** Refinar `dt` lo hace converger al solver en lo probado, pero esto no sustituye a una revisión del creador. El prototipo no es código de producción.

## Fuentes

- [Rojas et al., Robust VPINN](https://arxiv.org/pdf/2308.16910)
- [Lanthaler et al., Nonlinear reconstruction](https://arxiv.org/pdf/2210.01074)
- [PG-VarMiON, Petrov-Galerkin óptimo](https://arxiv.org/pdf/2503.04024)
- [Strong and weak residuals for advection-dominated diffusion](https://arxiv.org/pdf/2307.07647)
- [VINO](https://arxiv.org/pdf/2411.06587)
- [IONet](https://arxiv.org/pdf/2308.14537)
- [TFPONet](https://arxiv.org/pdf/2409.04782)
- [wPINN](https://arxiv.org/pdf/2207.08483)
- [McGreivy y Hakim, líneas base débiles](https://arxiv.org/pdf/2407.07218)
- [Zacchei et al., aceptación retardada multifidelidad](https://arxiv.org/abs/2512.16430)
- [Bakhos et al., contorno de Laplace para problemas parabólicos inversos](https://arxiv.org/pdf/1409.2556)
- [synax, simulador diferenciable de sincrotrón](https://arxiv.org/pdf/2410.01136)
- [Weak TransNet](https://arxiv.org/html/2506.14812)
