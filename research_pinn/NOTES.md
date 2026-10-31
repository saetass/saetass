# Formulación débil del transporte (notas de trabajo)

Variables adimensionales de `saetass.pinn.wind_bubble`: `x = r/R_TS`, velocidad en `v_b`, fuente de flujo unidad.

## Forma fuerte

    x² ∂ψ/∂t + ∂/∂x (x² J) = δ(x − 1),    J = v ψ − D ∂ψ/∂x

con `D` y `v` discontinuas en `x = 1` y `x = a`, y `ψ(ab) = 0`, `x²J → 0` en el origen.
En forma fuerte hay que imponer a mano: `[ψ] = 0`, `[x²J] = 1` en el choque, `[x²J] = 0` en `a`,
y la red debe producir segundas derivadas en capas de ancho `D/v`.

## Forma débil

Con `φ(ab) = 0`, integrando por partes:

    ∫ x² ∂ψ/∂t φ dx − ∫ x² (v ψ − D ∂ψ/∂x) ∂φ/∂x dx = φ(1)    ∀φ

Tres consecuencias:

1. La delta es `φ(1)`: no hay que suavizar la fuente.
2. Las condiciones de salto son naturales: desaparecen de la pérdida.
3. Sólo aparece `∂ψ/∂x`, no `∂²ψ/∂x²`.

## Qué norma: el punto clave

Sea `R(ψ̂)` el funcional residual. Si `L` es el operador, `R = L(ψ̂ − ψ)`, así que

    ‖ψ̂ − ψ‖_U ≈ ‖R‖_{V*}

**con la norma dual de `R`, no con la `L²`**. Minimizar `Σ R_k²` (VPINN) depende de la base de test y no acota
el error. Minimizar `‖R‖²_{V*} = Rᵀ G⁻¹ R`, con `G` la matriz de Gram de `V`, sí lo acota (RVPINN).

Tres elecciones de `G`, de menos a más informada:

- `G = I`: VPINN clásica.
- `G` = energía: `∫ x² D φ'φ' + (1/t) ∫ x² φφ`. Tridiagonal con funciones sombrero. Robusta en difusión.
- `G = (I − tA)` con `A` el operador Scharfetter–Gummel: es el precondicionador *óptimo* (test de Petrov-Galerkin
  óptimo discreto), porque Scharfetter–Gummel es un método con test exponencial. Debe ser estable a `Pe` alto.

Límite que hay que tener presente: con `G` exacta, `G⁻¹ R` **es** el error, luego la pérdida física equivale a la
de datos, pero sin datos previos y con una resolución por muestra (frente a 12-16 del contorno).

## Envolvente general

Para cualquier perfil estático, `ψ_ss = −A⁻¹F` cuesta una resolución tridiagonal real. También
`ψ_BE(t) = (I − tA)⁻¹ t F`, un paso implícito de tamaño `t`, una aproximación grosera de `ψ(t)` en una resolución.
La red aprende el cociente, y entra `ln ψ_env` como canal. Generaliza el truco de la burbuja a perfiles arbitrarios.

## Validación (r, p) con pérdidas (e5f.py)

`full_2d` con `ψ(0) = 0`, 400×71, `t = 0.05 Myr`, `ṗ = −B p²`. Saetass refinando `dt` converge al solver por contorno:
`N/Ninj` a `p = 3e5`: 0.4851 (dt), 0.4269 (dt/10), 0.4268 (dt/100), contorno 0.4268. L2 mediana: 9.1 %, 0.41 %, 0.35 %.
A `dt` del benchmark saetass da `N/Ninj = 1.015` a `p = 1e4`, imposible con escape: mecanismo de #74.
Falta: `ṗ` con cambio de signo (adiabáticas), para las que el barrido descendente no vale.

## Aceleración difusiva en choques (e9*.py)

Caso de `validation/cases/adiabatic_shock_acceleration.py`, `r_c = 4`, `ε = 0.02`. Estado estacionario directo: `A ψ + F = 0`,
matriz dispersa, Scharfetter–Gummel en `r` y upwind de 2.º orden en `log p` con el signo de `∇·v` en cada nodo.

| Método | Coste | Pendiente | Error frente a −2.0563 |
|---|---|---|---|
| saetass, 10 celdas | 17.7 s (8 hilos) | −2.0980 | 4.2e-2 |
| saetass, 20 celdas | 41.9 s (8 hilos) | −2.0775 | 2.1e-2 |
| directo, 20 celdas, 16 por década | 0.05 s (CPU) | −2.0658 | 9.5e-3 |
| directo, 20 celdas, 32 por década | 0.15 s | −2.0585 | 2.2e-3 |
| directo, 20 celdas, 128 por década | 1.5 s | −2.0565 | 2e-4 |

GMRES precondicionado con el barrido descendente: 47 iteraciones en 10 y en 20 celdas, independiente de la malla, por lo
que también es posible en GPU por lotes, sin factorizar. Con upwind de 1.er orden la pendiente sale −1.98: el orden en `p` importa.

Amplitud del espectro en el estacionario, razón a la amplitud plana a `p = 30 GeV/c`: método implícito 0.8977 (20 celdas, 64 por
década), 0.8978 (40 y 80 celdas). saetass: 0.8131 (20 celdas, 46 s), 0.8206 (40 celdas, 270 s), aún subiendo. Estimación manual:
corrección de pendiente `e^{−0.0565 ln(22.6)} = 0.838` por `3/(0.75−0.04)` frente a `3/0.75`, unos 0.885.
Contorno dependiente del tiempo con LU complejas: 1.2 s (16 por década) frente a 2-11 s de saetass; mismo 11 % de diferencia con
saetass a 20 celdas que en el estacionario, que atribuyo a su falta de convergencia, sin haberlo comprobado a tiempos cortos.

## Parámetros dependientes del tiempo (e6.py, e6b.py)

`D(t) = D f(t)`, `Q(t) = Q g(t)`, 256 nodos. Solver: TR-BDF2 por lotes en pasos logarítmicos. FNO con envolvente implícita congelada.
Convergencia del solver frente a 1920 pasos: 15 pasos 2 % (0.10 ms), 30 pasos 0.65 % / p95 7 % (0.20 ms), 60 pasos 0.2 % / p95 3 % (0.45 ms), 240 pasos 0.016 % / p95 0.5 % (1.9 ms).
FNO `be`, entrenado con etiquetas de 60 pasos: 1.0 % / p95 5 %, burbujas no vistas 2.0 % / p95 14 %, 0.22 ms. Empate alrededor del 1 %; por debajo gana el solver.

## Búsqueda de parámetros con el solver (e10.py)

Mismos datos que `pinn_parameter_search.py` (radios desde 16 pc). 512 caminantes, 400 pasos, malla de 1401 nodos, GPU compartida.
Medias: 38.3028, −4.2057, −22.4993, −1.0134; red: 38.3029, −4.2023, −22.4991, −1.0147. Mismas desviaciones. χ² con saetass 36.70 y 36.69.
Coste: 0.94 ms por perfil frente a 8.5 µs de la red. Punto de equilibrio: 3.6 h de entrenamiento equivalen a unos 14 millones de perfiles de solver.

## Evaluación sin malla (e11.py)

Interpolación logarítmica a 4000 radios aleatorios frente a resolución 8: L2 mediano 1.6e-5 con 1401 nodos, peor 1.1e-3, máximo puntual 1.01 (colas). Con 5601 nodos, peor 5.4e-5 y máximo 0.04.

## Contraste de D sin reentrenar (e12.py)

Modelos entrenados con saltos hasta ×30. L2 mediano y p95 en burbujas con contraste en `R_b`:
`none_data`: ×100 10.9/32 %, ×1e4 30.8/52 %, ×1e6 63/82 %. `be_data`: ×100 2.1/5.1 %, ×1e4 2.9/7.4 %, ×1e6 5.4/13.9 %.
`ss_data` y `ss_both` mantienen la mediana (7.3 % y 10.6 % a ×1e6) pero con colas de 72-82 %.

## Operador 2D (r, p) con pérdidas (e7.py)

128×48, 12000 pasos. Con envolvente estacionaria: L2 0.77 % / p95 7.5 %, burbujas no vistas 1.0 % / p95 10 %; espectro 0.38 % y 0.59 %.
Sin envolvente: 3.5 % / 13 %, burbujas 8.6 % / 17 %. Coste: solver 4.3 ms, envolvente 0.39 ms, red 1.26 ms por muestra.

## Línea base: contorno con pocos nodos (e13.py)

256 nodos, 2048 perfiles, GPU compartida. Burbujas: N=2 12.8 % (6 µs), N=4 0.8 % (9 µs), N=6 0.06 % (15 µs), N=8 0.004 % (19 µs), N=12 1.7e-7 (30 µs).
Perfiles aleatorios: igual en mediana, pero hay casos atípicos.

Casos atípicos (e13b.py, 8192 perfiles, referencia TR-BDF2 de 400 pasos): con `Pe` de celda < 1, 0 fallos en 7512. Con `Pe` de celda 1-10, 1.46 % sobre 1 %; 10-100, 8 %.
Causa: operador muy no normal cuando la malla no resuelve la capa. El escalado simétrico no lo cura.
Detector (e13c.py): principio del máximo `0 ≤ ψ ≤ ψ_ss`. N=24, tolerancia 1e-3: marca 0.31 %, atrapa 14 de 14, peor no marcado 0.43 %. N=16: atrapa 37 de 40, peor no marcado 2.8 %. Sólo vale con fuente constante y coeficientes estáticos.

## Benchmarks extremos (e14.py)

`protons_2d`, 148×400, `t = 2e-3`, pérdidas reales, Pe de celda máx 0.44: saetass refinando `dt` converge al contorno. L2 por `p`, mediana: 18 % (dt/4), 5.4 % (dt/16), 0.0 % (dt/64), p95 2.5 %. Espectro: 0 % de mediana, p95 1.2 %. Contorno 3.5 s.
`electrons_2d`, 101×400, `t = 5e-4`, Bohm, Pe de celda 1.25e8: NO converge. L2 mediana 8.5 %, p95 100 %, sin mejora de dt/4 a dt/64. Causa probable, sin comprobar: Scharfetter–Gummel es upwind de 1.er orden a Pe de celda grande.

## Tiempos con GPU libre (e15.py, e10.py)

256 nodos, lote 1024: resolución real 1.6 µs; contorno N=4 3.7 µs, N=6 5.4 µs, N=8 7.2 µs, N=16 14 µs, N=24 21 µs; pasada FNO 35 µs.
Búsqueda con solver, 1401 nodos, N=12: 0.102 ms por perfil, mismo posterior que con N=24. Red: 8.5 µs. Equilibrio: 3.6 h / 0.1 ms ≈ 130 millones de perfiles.

## Segunda ronda: ventaja publicable

Literatura: no aparece solver diferenciable de transporte de rayos cósmicos (lo más cercano, synax, es de sincrotrón). Contorno de Laplace para PDE parabólicas inversas: Bakhos et al., con Krylov, sin lotes GPU ni volúmenes finitos advectivos. Aceptación retardada multifidelidad ya existe como método general.
Modelo genérico `gen.Forward`: cualquier `D(r)`, `v(r)`, malla uniforme. χ² en la verdad (saetass 37.4): 4096 nodos 39.3, 2048 40.0, 1024 37.3, 512 67.7, 256 106.7, 128 322.
Grueso a fino con discrepancia lineal e importancia: a 128 nodos ESS = 1 de 1024, la corrección exacta colapsa.
Gradiente por autodiferenciación del contorno: coincide con diferencias finitas (8.628021e+03).

Saetass como etapa final (e20.py): posterior GPU (1024 nodos, 176 s con GPU compartida), corregido por importancia con 256 puntos × 4 energías = 1024 corridas reales (1200 celdas), 281 s en 16 núcleos. ESS 214 de 256. Medias 38.3030, −4.198, −22.4992, −1.0142, a ≤0.7σ de la verdad. Directo equivaldría a 820 000 perfiles × 4.3 s ≈ 980 h·núcleo.
Inversión de `D(r)` (e19.py): 40 nodos uniformes, 16 realizaciones, ruido 5 %, 40 radios, 4 energías, 484 s. χ²/N 1.08. |Δln D| mediano 0.026 (viento), 0.022 (burbuja), 0.024 (medio). Máximo 3.2 cerca de `R_b`. Cobertura de la banda 68 %: 0.08, 0.80, 0.78: el viento está sobreconfiado.
FNO genérico como modelo burdo (e21.py), burbuja física fuera de su rango: rms 0.117 (0.052 sin sesgo medio), peor que 256 nodos (0.045).
Inversión con 160 nodos (e19b.py), 650 s: χ²/N 0.99. |Δln D| mediano 0.023, 0.023, 0.024; a 92-120 pc 0.03 (0.22 con 40 nodos). Cobertura 0.28, 0.91, 0.95. El salto de `R_b` cae ~5 pc desplazado, comparable a la separación de los radios de datos (~9 pc). Pesa `r < 3 pc`, sin datos: mediana 0.25.
