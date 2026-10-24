"""
Algorithms Optimization - Tarea: Ejercicios 1 a 5
Usa los codigos de clase (estimate_pi, knapsack, 2-opt, SA, mst_prim/preorder).

Uso:   python ejercicios_algoritmia.py          (corre los 5 ejercicios)
       python ejercicios_algoritmia.py 3        (corre solo el ejercicio 3)
       python ejercicios_algoritmia.py 1 4      (corre el 1 y el 4)
Las graficas se guardan en OUTPUT_DIR.
"""
import math
import os
import random
import statistics as st
import sys
import time
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt

OUTPUT_DIR = "graficas"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, name), dpi=150)
    plt.show()


# ======================================================================
# Funciones compartidas (las de clase, con pequenos agregos)
# ======================================================================
def tour_length(points, route):
    return sum(math.dist(points[route[i]], points[route[(i + 1) % len(route)]])
               for i in range(len(route)))


def two_opt_move(route, i, k):
    """Reverse route[i..k]: removes edges (i-1, i) and (k, k+1)."""
    return route[:i] + route[i:k + 1][::-1] + route[k + 1:]


def two_opt_local_search(points, route):
    """2-opt como hill climbing: repite el movimiento para cada par (i, k)
    hasta que ninguna reversion acorta el tour (optimo local).
    Regresa (ruta final, numero de mejoras)."""
    route = route[:]
    best_len = tour_length(points, route)
    improvements = 0
    improved = True
    while improved:
        improved = False
        for i in range(1, len(route) - 1):
            for k in range(i + 1, len(route)):
                candidate = two_opt_move(route, i, k)
                cand_len = tour_length(points, candidate)
                if cand_len < best_len - 1e-12:
                    route, best_len = candidate, cand_len
                    improvements += 1
                    improved = True
    return route, improvements


def nearest_neighbour(points, start=0):
    """Heuristica constructiva: ir siempre a la ciudad no visitada mas cercana."""
    unvisited = set(range(len(points))) - {start}
    route = [start]
    while unvisited:
        last = route[-1]
        nxt = min(unvisited, key=lambda c: math.dist(points[last], points[c]))
        route.append(nxt)
        unvisited.remove(nxt)
    return route


def random_route(n, rng):
    route = list(range(n))
    rng.shuffle(route)
    return route


def random_points(n, rng, size=100.0):
    return [(rng.uniform(0, size), rng.uniform(0, size)) for _ in range(n)]


def held_karp(points):
    """Longitud del tour OPTIMO (programacion dinamica, O(n^2 2^n)).
    Sirve de referencia para C_optimo en los ejercicios 3 y 5."""
    n = len(points)
    d = [[math.dist(points[a], points[b]) for b in range(n)] for a in range(n)]
    INF = float("inf")
    dp = [[INF] * n for _ in range(1 << n)]
    dp[1][0] = 0.0
    for mask in range(1 << n):
        if not mask & 1:
            continue
        for last in range(n):
            cur = dp[mask][last]
            if cur == INF:
                continue
            for nxt in range(n):
                if (mask >> nxt) & 1:
                    continue
                nm = mask | (1 << nxt)
                c = cur + d[last][nxt]
                if c < dp[nm][nxt]:
                    dp[nm][nxt] = c
    full = (1 << n) - 1
    return min(dp[full][j] + d[j][0] for j in range(1, n))


def mst_prim(points):
    """Prim desde la ciudad 0; regresa el arbol como {nodo: hijos}."""
    best = {v: (math.dist(points[0], points[v]), 0) for v in range(1, len(points))}
    children = {v: [] for v in range(len(points))}
    while best:
        v = min(best, key=lambda u: best[u][0])
        children[best.pop(v)[1]].append(v)
        for u in best:
            d = math.dist(points[v], points[u])
            if d < best[u][0]:
                best[u] = (d, v)
    return children


def preorder(children, root=0):
    """Recorrido del arbol visitando cada ciudad una vez = atajos."""
    return [root] + [c for child in children[root] for c in preorder(children, child)]


def mst_approximation(points):
    return preorder(mst_prim(points))


# ---- Simulated Annealing (pseudocodigo de clase) ----------------------
def simulated_annealing(points, initial_route, schedule, rng, record=False):
    """function SIMULATED-ANNEALING(problem, schedule) returns a state
    Minimizacion: delta_E = L(actual) - L(siguiente)  (>0 => el siguiente es mejor).
    Movimiento = reversion 2-opt al azar."""
    current = initial_route[:]
    cur_len = tour_length(points, current)
    n = len(points)
    costs, temps = [cur_len], [None]
    t = 1
    while True:
        T = schedule(t)
        if T == 0:                                   # if T = 0 then return current
            return current, cur_len, costs, temps
        i, k = sorted(rng.sample(range(1, n), 2))    # Diff 1: vecino al azar
        nxt = two_opt_move(current, i, k)
        nxt_len = tour_length(points, nxt)
        delta_E = cur_len - nxt_len
        if delta_E > 0 or rng.random() < math.exp(delta_E / T):     # Diff 2: acepta malos
            current, cur_len = nxt, nxt_len
        if record:
            costs.append(cur_len)
            temps.append(T)
        t += 1


# Esquemas de enfriamiento: todos van de T0 a T_min en N iteraciones y
# regresan 0 despues (asi el algoritmo termina como en el pseudocodigo).
def linear_schedule(T0, Tmin, N):
    return lambda t: 0 if t > N else T0 - (T0 - Tmin) * t / N


def exponential_schedule(T0, Tmin, N):
    alpha = (Tmin / T0) ** (1 / N)
    return lambda t: 0 if t > N else T0 * alpha ** t


def logarithmic_schedule(T0, Tmin, N):
    c = (T0 / Tmin - 1) / math.log(1 + N)
    return lambda t: 0 if t > N else T0 / (1 + c * math.log(1 + t))


SCHEDULES = {"Lineal": linear_schedule,
             "Exponencial": exponential_schedule,
             "Logaritmico": logarithmic_schedule}


# ======================================================================
# EJERCICIO 1 - Parallel Speedup
# ======================================================================
def estimate_pi(seed: int, samples: int = 1_000_000) -> float:
    """Monte Carlo: fraction of random points that fall inside the unit quarter circle."""
    rng = random.Random(seed)
    inside = sum(rng.random() ** 2 + rng.random() ** 2 <= 1.0 for _ in range(samples))
    return 4 * inside / samples


def exercise1(repeats=2, include_threads=True):
    print("\n" + "=" * 72 + "\nEJERCICIO 1 - Aceleracion paralela (Parallel Speedup)\n" + "=" * 72)
    cores = mp.cpu_count()
    print(f"Nucleos logicos detectados: {cores}")
    seeds = list(range(8))                           # 8 corridas independientes
    procs = [1, 2, 4, 8]

    def best_of(fn):
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t0)
        return min(times)

    t_seq = best_of(lambda: [estimate_pi(s) for s in seeds])

    def run_pool(p):
        with mp.Pool(processes=p) as pool:
            pool.map(estimate_pi, seeds)

    def run_threads(p):
        with ThreadPoolExecutor(max_workers=p) as ex:
            list(ex.map(estimate_pi, seeds))

    t_proc = {p: best_of(lambda p=p: run_pool(p)) for p in procs}
    t_thr = {p: best_of(lambda p=p: run_threads(p)) for p in procs} if include_threads else {}

    S_proc = {p: t_seq / t_proc[p] for p in procs}
    S_thr = {p: t_seq / t_thr[p] for p in procs} if include_threads else {}

    # Ley de Amdahl: S(p) = 1 / (f + (1-f)/p). Se estima la fraccion NO paralelizable f
    # con el speedup medido en 8 procesos: f = (1/S - 1/p) / (1 - 1/p)
    p_ref = procs[-1]
    f_serial = max(0.0, min(1.0, (1 / S_proc[p_ref] - 1 / p_ref) / (1 - 1 / p_ref)))

    print(f"\nTiempo secuencial (loop, sin pool): {t_seq:.3f} s")
    header = f"{'procesos':>9} | {'tiempo (s)':>10} {'speedup':>8} {'eficiencia':>11}"
    if include_threads:
        header += f" | {'threads (s)':>11} {'speedup':>8}"
    print(header)
    for p in procs:
        row = f"{p:>9} | {t_proc[p]:>10.3f} {S_proc[p]:>7.2f}x {S_proc[p] / p * 100:>10.0f}%"
        if include_threads:
            row += f" | {t_thr[p]:>11.3f} {S_thr[p]:>7.2f}x"
        print(row)
    print(f"\nFraccion no paralelizable estimada (Amdahl, con p={p_ref}): f = {f_serial:.3f}")
    print(f"Speedup maximo teorico con esa f (p -> infinito): {1 / f_serial:.1f}x"
          if f_serial > 0 else "f ~ 0: Amdahl no limita el speedup en este rango.")

    xs = [1, 2, 3, 4, 5, 6, 7, 8]
    plt.figure(figsize=(7.5, 5))
    plt.plot(procs, procs, "k--", label="Speedup ideal (lineal)")
    plt.plot(xs, [1 / (f_serial + (1 - f_serial) / p) for p in xs], ":", color="gray",
             label=f"Ley de Amdahl (f = {f_serial:.2f})")
    plt.plot(procs, [S_proc[p] for p in procs], "o-", lw=2, label="multiprocessing (medido)")
    if include_threads:
        plt.plot(procs, [S_thr[p] for p in procs], "s--", label="threading (medido, extra)")
    plt.xlabel("Numero de procesos")
    plt.ylabel("Speedup  (t secuencial / t paralelo)")
    plt.title("Ejercicio 1: speedup de estimate_pi (8 corridas x 1,000,000 muestras)")
    plt.xticks(procs)
    plt.grid(alpha=0.3)
    plt.legend()
    save("ej1_speedup.png")


# ======================================================================
# EJERCICIO 2 - Greedy knapsack: menor peso primero vs valor/peso
# ======================================================================
def greedy_knapsack(values, weights, W, key):
    """Greedy generico: ordena los indices con `key` y mete items mientras quepan."""
    total_v = total_w = 0
    chosen = []
    for i in sorted(range(len(values)), key=key):
        if total_w + weights[i] <= W:
            chosen.append(i)
            total_w += weights[i]
            total_v += values[i]
    return total_v, total_w, sorted(chosen)


def knapsack_dp(values, weights, W):
    n = len(values)
    K = [[0] * (W + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        v, w = values[i - 1], weights[i - 1]
        for c in range(W + 1):
            K[i][c] = K[i - 1][c]
            if w <= c:
                K[i][c] = max(K[i][c], v + K[i - 1][c - w])
    return K[n][W]


def compare_knapsack(title, names, values, weights, W):
    methods = {
        "Menor peso primero": greedy_knapsack(values, weights, W, key=lambda i: weights[i]),
        "Valor/peso": greedy_knapsack(values, weights, W, key=lambda i: -values[i] / weights[i]),
    }
    opt = knapsack_dp(values, weights, W)
    print(f"\n--- {title} (capacidad W = {W}) ---")
    print("  Objetos: " + ", ".join(f"{n}(v={v}, w={w}, v/w={v / w:.2f})"
                                    for n, v, w in zip(names, values, weights)))
    print(f"  {'Metodo':<20}{'Objetos elegidos':<20}{'Capacidad usada':>16}{'Valor total':>13}")
    for label, (v, w, ch) in methods.items():
        print(f"  {label:<20}{str([names[i] for i in ch]):<20}{f'{w}/{W}':>16}{v:>13}")
    print(f"  {'Optimo (DP)':<20}{'':<20}{'':>16}{opt:>13}")
    return methods, opt


def exercise2():
    print("\n" + "=" * 72 + "\nEJERCICIO 2 - Greedy knapsack: menor peso primero\n" + "=" * 72)
    # Items del PDF (referencia)
    compare_knapsack("Items del PDF", ["A", "B", "C"], [60, 100, 120], [10, 20, 30], 50)

    # Instancia disenada: menor peso primero obtiene un resultado PEOR que valor/peso
    names, values, weights, W = ["A", "B", "C", "D"], [1, 1, 1, 50], [1, 2, 3, 5], 10
    methods, opt = compare_knapsack("Instancia disenada", names, values, weights, W)
    lw, vr = methods["Menor peso primero"][0], methods["Valor/peso"][0]
    assert lw < vr
    print(f"\n  Menor peso primero = {lw}  <  valor/peso = {vr}  (optimo = {opt})")

    # Grafica de barras + tabla
    labels = list(methods)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), gridspec_kw={"width_ratios": [1, 1, 1.4]})
    colors = ["tab:red", "tab:green"]
    axes[0].bar(labels, [methods[m][0] for m in labels], color=colors)
    axes[0].axhline(opt, color="k", linestyle="--", label=f"Optimo = {opt}")
    axes[0].set_title("Valor total")
    axes[0].legend()
    axes[1].bar(labels, [methods[m][1] for m in labels], color=colors)
    axes[1].axhline(W, color="k", linestyle="--", label=f"Capacidad = {W}")
    axes[1].set_title("Capacidad usada")
    axes[1].legend()
    for ax in axes[:2]:
        ax.tick_params(axis="x", labelrotation=10)
        for bar in ax.patches:
            ax.annotate(f"{bar.get_height():g}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        ha="center", va="bottom")
    axes[2].axis("off")
    rows = [[m, ", ".join(names[i] for i in methods[m][2]), f"{methods[m][1]}/{W}", methods[m][0]]
            for m in labels]
    table = axes[2].table(cellText=rows, colLabels=["Metodo", "Objetos", "Capacidad", "Valor"],
                          loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    axes[2].set_title("Objetos seleccionados")
    plt.suptitle("Ejercicio 2: instancia A(1,1) B(1,2) C(1,3) D(50,5), W = 10   (objeto = valor, peso)")
    save("ej2_knapsack.png")


# ======================================================================
# EJERCICIO 3 - 2-opt: inicio aleatorio vs Vecino Mas Cercano
# ======================================================================
def exercise3(n_instances=200, n_cities=10):
    print("\n" + "=" * 72 + "\nEJERCICIO 3 - Busqueda local 2-opt\n" + "=" * 72)
    rng = random.Random(2024)
    imp_rand, imp_nn, gap_rand, gap_nn, start_len_rand, start_len_nn = [], [], [], [], [], []
    for _ in range(n_instances):
        pts = random_points(n_cities, rng)
        opt = held_karp(pts)
        r0 = random_route(n_cities, rng)                                  # inicio aleatorio
        nn0 = nearest_neighbour(pts, start=rng.randrange(n_cities))       # NN desde nodo aleatorio
        rf, ri = two_opt_local_search(pts, r0)
        nf, ni = two_opt_local_search(pts, nn0)
        imp_rand.append(ri)
        imp_nn.append(ni)
        gap_rand.append((tour_length(pts, rf) / opt - 1) * 100)
        gap_nn.append((tour_length(pts, nf) / opt - 1) * 100)
        start_len_rand.append(tour_length(pts, r0) / opt)
        start_len_nn.append(tour_length(pts, nn0) / opt)

    print(f"{n_instances} instancias aleatorias de {n_cities} ciudades (optimo por Held-Karp)\n")
    print(f"{'':<26}{'Inicio aleatorio':>18}{'Inicio NN':>12}")
    print(f"{'Costo inicial / optimo':<26}{st.mean(start_len_rand):>18.2f}{st.mean(start_len_nn):>12.2f}")
    print(f"{'Mejoras (media)':<26}{st.mean(imp_rand):>18.2f}{st.mean(imp_nn):>12.2f}")
    print(f"{'Mejoras (mediana)':<26}{st.median(imp_rand):>18.1f}{st.median(imp_nn):>12.1f}")
    print(f"{'Mejoras (max)':<26}{max(imp_rand):>18}{max(imp_nn):>12}")
    print(f"{'Gap del optimo local (%)':<26}{st.mean(gap_rand):>18.2f}{st.mean(gap_nn):>12.2f}")
    print(f"{'Veces que llega al optimo':<26}{sum(g < 1e-9 for g in gap_rand):>14}/{n_instances}"
          f"{sum(g < 1e-9 for g in gap_nn):>8}/{n_instances}")
    fewer = sum(b < a for a, b in zip(imp_rand, imp_nn))
    equal = sum(b == a for a, b in zip(imp_rand, imp_nn))
    print(f"\nNN necesita MENOS mejoras en {fewer}/{n_instances} instancias "
          f"(igual en {equal}, mas en {n_instances - fewer - equal}).")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].boxplot([imp_rand, imp_nn], tick_labels=["Aleatorio", "Vecino mas cercano"], showmeans=True)
    axes[0].set_ylabel("Numero de mejoras 2-opt")
    axes[0].set_title("Mejoras necesarias hasta el optimo local")
    axes[1].boxplot([gap_rand, gap_nn], tick_labels=["Aleatorio", "Vecino mas cercano"], showmeans=True)
    axes[1].set_ylabel("Gap respecto al optimo (%)")
    axes[1].set_title("Calidad del optimo local alcanzado")
    for ax in axes:
        ax.grid(alpha=0.3)
    plt.suptitle(f"Ejercicio 3: 2-opt en {n_instances} instancias de {n_cities} ciudades (triangulo = media)")
    save("ej3_boxplots.png")


# ======================================================================
# EJERCICIO 4 - Simulated Annealing: esquemas de enfriamiento
# ======================================================================
def exercise4(n_cities=30, runs=8, T0=100.0, Tmin=0.1):
    print("\n" + "=" * 72 + "\nEJERCICIO 4 - Recocido Simulado (cooling schedules)\n" + "=" * 72)
    rng_inst = random.Random(11)
    points = random_points(n_cities, rng_inst)
    start = random_route(n_cities, random.Random(5))             # solucion inicial aleatoria
    print(f"Instancia de {n_cities} ciudades. Costo de la ruta inicial aleatoria: "
          f"{tour_length(points, start):.1f}.  T0 = {T0:g}, Tmin = {Tmin:g}")

    # --- (1) Curvas de convergencia (mismo presupuesto N para los tres esquemas) ---
    N_conv = 20_000
    curves = {}
    for name, factory in SCHEDULES.items():
        runs_costs = []
        for r in range(runs):
            rng = random.Random(100 + r)
            _, _, costs, temps = simulated_annealing(points, start, factory(T0, Tmin, N_conv), rng, record=True)
            runs_costs.append((costs, temps))
        curves[name] = runs_costs

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    for name, runs_costs in curves.items():
        mean_cost = [st.mean(rc[0][i] for rc in runs_costs) for i in range(0, N_conv + 1, 50)]
        axes[0].plot(range(0, N_conv + 1, 50), mean_cost, label=name)
        costs0, temps0 = runs_costs[0]
        axes[1].plot(temps0[1::25], costs0[1::25], lw=0.8, label=name)
    axes[0].set_xlabel("Iteracion")
    axes[0].set_ylabel("Costo (media de las corridas)")
    axes[0].set_title("Costo vs iteraciones")
    axes[1].set_xscale("log")
    axes[1].invert_xaxis()
    axes[1].set_xlabel("Temperatura (log, enfria hacia la derecha)")
    axes[1].set_ylabel("Costo (una corrida)")
    axes[1].set_title("Costo vs temperatura")
    # perfil de temperatura de cada esquema
    ts = list(range(1, N_conv + 1, 20))
    for name, factory in SCHEDULES.items():
        sch = factory(T0, Tmin, N_conv)
        axes[2].plot(ts, [sch(t) for t in ts], label=name)
    axes[2].set_xlabel("Iteracion")
    axes[2].set_ylabel("Temperatura")
    axes[2].set_title("Perfil de temperatura de cada esquema")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
    plt.suptitle(f"Ejercicio 4: curvas de convergencia (N = {N_conv:,} iteraciones por corrida)")
    save("ej4_convergencia.png")

    # --- (2) Tiempo de ejecucion vs calidad (variando el presupuesto N) ---
    budgets = [1_000, 3_000, 10_000, 30_000]
    results = {name: [] for name in SCHEDULES}          # (N, tiempo medio, costo final medio)
    for name, factory in SCHEDULES.items():
        for N in budgets:
            times, finals = [], []
            for r in range(runs):
                rng = random.Random(500 + r)
                t0 = time.perf_counter()
                _, cost, _, _ = simulated_annealing(points, start, factory(T0, Tmin, N), rng)
                times.append(time.perf_counter() - t0)
                finals.append(cost)
            results[name].append((N, st.mean(times), st.mean(finals)))
    best_known = min(r[2] for rs in results.values() for r in rs)

    print(f"\nCosto final promedio ({runs} corridas) y tiempo promedio por esquema y presupuesto N")
    print(f"(gap = respecto al mejor costo promedio encontrado: {best_known:.1f})")
    print(f"{'Esquema':<13}{'N':>8}{'costo final':>13}{'gap %':>8}{'tiempo (s)':>12}")
    for name, rs in results.items():
        for N, tm, c in rs:
            print(f"{name:<13}{N:>8,}{c:>13.1f}{(c / best_known - 1) * 100:>8.1f}{tm:>12.3f}")

    plt.figure(figsize=(8, 5.5))
    for name, rs in results.items():
        xs, ys = [r[1] for r in rs], [r[2] for r in rs]
        plt.plot(xs, ys, "o-", label=name)
        for N, tm, c in rs:
            plt.annotate(f"{N // 1000}k", (tm, c), textcoords="offset points", xytext=(4, 4), fontsize=8)
    plt.xscale("log")
    plt.xlabel("Tiempo de ejecucion (s, escala log)")
    plt.ylabel("Costo final promedio (menor es mejor)")
    plt.title("Ejercicio 4: tiempo vs calidad (etiquetas = iteraciones N)")
    plt.grid(alpha=0.3)
    plt.legend()
    save("ej4_tiempo_vs_calidad.png")


# ======================================================================
# EJERCICIO 5 - Razon de aproximacion (MST) sobre 100 instancias de 9 ciudades
# ======================================================================
def exercise5(n_instances=100, n_cities=9, N_sa=3_000):
    print("\n" + "=" * 72 + "\nEJERCICIO 5 - Razon de aproximacion (MST)\n" + "=" * 72)
    costs = {"MST puro": [], "MST + 2-opt": [], "Busqueda local (2-opt)": [], "Recocido simulado": []}
    optimal = []
    for s in range(n_instances):
        rng = random.Random(s)
        pts = random_points(n_cities, rng, size=10.0)
        opt = held_karp(pts)
        optimal.append(opt)

        approx = mst_approximation(pts)
        costs["MST puro"].append(tour_length(pts, approx))
        costs["MST + 2-opt"].append(tour_length(pts, two_opt_local_search(pts, approx)[0]))

        r0 = random_route(n_cities, rng)
        costs["Busqueda local (2-opt)"].append(tour_length(pts, two_opt_local_search(pts, r0)[0]))

        sch = exponential_schedule(T0=5.0, Tmin=0.01, N=N_sa)
        _, sa_cost, _, _ = simulated_annealing(pts, r0, sch, rng)
        costs["Recocido simulado"].append(sa_cost)

    ratios = {k: [c / o for c, o in zip(v, optimal)] for k, v in costs.items()}
    mst_r = ratios["MST puro"]

    print(f"{n_instances} instancias aleatorias de {n_cities} ciudades (optimo por Held-Karp)\n")
    print("Razon C_obtenido / C_optimo de la aproximacion MST:")
    print(f"  media = {st.mean(mst_r):.3f}   mediana = {st.median(mst_r):.3f}   "
          f"maxima = {max(mst_r):.3f}   minima = {min(mst_r):.3f}   (limite teorico = 2)")
    print(f"  la peor instancia usa {(max(mst_r) - 1) / (2 - 1) * 100:.0f}% del margen permitido "
          f"(entre 1 y 2); ninguna supera 2: {all(r <= 2 + 1e-9 for r in mst_r)}")
    print("\nComparacion de metodos (razon C / C_optimo):")
    print(f"  {'Metodo':<24}{'media':>8}{'mediana':>9}{'maxima':>8}{'optimo exacto':>16}")
    for k, v in ratios.items():
        print(f"  {k:<24}{st.mean(v):>8.3f}{st.median(v):>9.3f}{max(v):>8.3f}"
              f"{sum(r < 1 + 1e-9 for r in v):>12}/{n_instances}")
    gain = (st.mean(mst_r) - st.mean(ratios['MST + 2-opt'])) / (st.mean(mst_r) - 1) * 100
    print(f"\n2-opt como post-procesamiento elimina {gain:.0f}% del exceso de la aproximacion MST sobre el optimo.")

    # (1) Histograma + dispersion de la razon MST con linea en 2
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].hist(mst_r, bins=15, color="tab:blue", edgecolor="white")
    axes[0].axvline(2, color="r", linestyle="--", label="Limite teorico = 2")
    axes[0].axvline(st.mean(mst_r), color="k", linestyle=":", label=f"Media = {st.mean(mst_r):.2f}")
    axes[0].set_xlim(0.95, 2.1)
    axes[0].set_xlabel("C_obtenido / C_optimo")
    axes[0].set_ylabel("Numero de instancias")
    axes[0].set_title("Histograma de la razon de aproximacion")
    axes[1].scatter(range(1, n_instances + 1), mst_r, s=14)
    axes[1].axhline(2, color="r", linestyle="--", label="Limite teorico = 2")
    axes[1].axhline(1, color="k", linestyle=":", label="Optimo = 1")
    axes[1].set_ylim(0.95, 2.1)
    axes[1].set_xlabel("Instancia")
    axes[1].set_ylabel("C_obtenido / C_optimo")
    axes[1].set_title("Dispersion por instancia")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.3)
    plt.suptitle("Ejercicio 5: aproximacion MST en 100 instancias de 9 ciudades")
    save("ej5_razon_mst.png")

    # (2) Boxplots comparativos: costo y razon
    names = list(costs)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].boxplot([costs[k] for k in names] + [optimal], tick_labels=names + ["Optimo"], showmeans=True)
    axes[0].set_ylabel("Costo del tour")
    axes[0].set_title("Costo obtenido")
    axes[1].boxplot([ratios[k] for k in names], tick_labels=names, showmeans=True)
    axes[1].axhline(2, color="r", linestyle="--", label="Limite teorico = 2")
    axes[1].axhline(1, color="k", linestyle=":", label="Optimo = 1")
    axes[1].set_ylabel("C_obtenido / C_optimo")
    axes[1].set_title("Costo normalizado por el optimo")
    axes[1].legend()
    for ax in axes:
        ax.tick_params(axis="x", labelrotation=15)
        ax.grid(alpha=0.3)
    plt.suptitle("Ejercicio 5: MST puro, MST + 2-opt, busqueda local y recocido simulado")
    save("ej5_comparacion_metodos.png")


# ======================================================================
if __name__ == "__main__":
    exercises = {"1": exercise1, "2": exercise2, "3": exercise3, "4": exercise4, "5": exercise5}
    for key in (sys.argv[1:] or list(exercises)):
        exercises[key]()