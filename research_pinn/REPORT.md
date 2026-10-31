# Cómo reproducir saetass en GPU: PINN, formulación débil, operadores neuronales y solvers

Investigación del 31 de octubre de 2026, rama `PINN`, sin commits. Código en este directorio y derivaciones en [NOTES.md](NOTES.md).

## Conclusión

**La formulación débil funciona y arregla lo que fallaba, pero la mejor opción para saetass no es una red: es la propia formulación débil resuelta directamente en GPU.** La ecuación es lineal, y eso permite obtener `ψ(t)` con unas pocas resoluciones tridiagonales, sin pasos de tiempo. Las redes solo merecen la pena en un nicho.

| Pregunta | Respuesta |
|---|---|
| ¿Arregla la forma débil las discontinuidades? | Sí, si se usa la norma dual con test exponencial. La VPINN clásica y el residuo fuerte fallan. |
| ¿Sirve un operador neuronal para discontinuidades fuertes? | Sí, con envolvente numérica: 5,4 % con contraste ×10⁶, frente a 63 % sin ella. |
| ¿Se puede entrenar sin datos? | En un solo problema, sí (0,13-1,2 %). En un operador general, no con estas pruebas. |
| ¿Gana alguna red al solver? | En estático, no. En perfiles temporales, empate al 1 %. En muestreo masivo de una familia fija, sí, unas 12 veces. |
| ¿Cubre el solver a saetass? | En casi todo lo probado. No en advección pura sin resolver (Bohm, `Pe` de celda ≫ 1). |

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
