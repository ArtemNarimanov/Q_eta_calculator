#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Плоское ребро: расчёт мощности теплоотдачи Q̇ и эффективности η_f
по численному профилю температуры T(x).

Физическая модель
-----------------
Плоское прямоугольное ребро ширины b и высоты охлаждаемой части L.
Обе широкие поверхности охлаждаются воздухом при температуре T1 с
коэффициентом теплоотдачи α. Охлаждаемый периметр P = 2b, площадь
теплоотдачи A_f = P·L = 2bL. Торец (x = L) и узкие кромки изолированы.

По численному профилю T(x), x ∈ [0, L], вычисляются:

    I      = ∫[0..L] (T(x) − T1) dx        [К·м]
    Q̇     = α · P · I                     [Вт]
    η_f   = I / (L · (T0 − T1))            [-]   (α сокращается — η_f от α не зависит)

Перекрёстная проверка: подгонка M из профиля θ(x) = A·cosh[M·(1 − x/L)]
(A — по наименьшим квадратам, M — методом золотого сечения) и аналитическая
эффективность η_analytic = tanh(M)/M.

Приложение НЕ решает задачу теплопроводности: λ и δ не требуются.
Вход: b, L, α, T0, T1 + два .dat-файла (x и T).

Только стандартная библиотека Python 3.9+ (math, pathlib, tkinter) —
портативно, офлайн, без установки и админских прав. UI на русском языке.
"""

from __future__ import annotations

import math
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import tkinter as tk
from tkinter import filedialog, messagebox

# ----------------------------- константы -----------------------------

DEFAULT_B = 0.100        # ширина ребра, м
DEFAULT_L = 0.060        # высота охлаждаемой части, м
DEFAULT_ALPHA = 15.0     # коэффициент теплоотдачи, Вт/(м²·К)
DEFAULT_T0_C = 120.0     # температура основания, °C
DEFAULT_T1_C = 20.0      # температура воздуха, °C

KELVIN_OFFSET = 273.15   # K = °C + 273.15
TOL_DT_EQ = 1e-12        # |T0 − T1| меньше этого считается нулём (особый случай)
M_FIT_LO = 0.05          # интервал поиска M (золотое сечение): нижняя граница
M_FIT_HI = 20.0          # ... верхняя граница
M_FIT_TOL = 1e-10        # конечная ширина интервала при сходимости
FIT_WARN_REL_RES = 0.05  # порог «плохой подгонки M» — относительный остаток > 5 %
RANGE_TOL_FACTOR = 1e-9  # допуск x вне [0, L]: tol = 1e-9·L

# Числовой токен: 0.01, 1e-3, -2.5, +3.14, .5, 3.
# Lookbehind исключает цифры, «вшитые» в слова (например, «LAB9» в заголовке).
NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_.])([-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?)")

# Кандидатные кодировки входных файлов (по порядку).
ENCODINGS = ("utf-8-sig", "cp1251", "latin-1")

# Уровни строк лога.
LVL_INFO = "INFO"
LVL_WARN = "WARNING"
LVL_ERROR = "ERROR"


# --------------------------- вспомогательные ---------------------------

def _log(log: Optional[list], level: str, msg: str) -> None:
    """Добавить строку в лог (список строк), если он передан."""
    if log is not None:
        log.append(f"{level}: {msg}")


def fmt4(v) -> str:
    """Формат числа с 4 значащими цифрами; None → «—»."""
    if v is None:
        return "—"
    return f"{v:.4g}"


# --------------------------- чтение .dat-файлов ---------------------------

def _read_lines(path: Path, log: Optional[list]) -> List[str]:
    """Прочитать текстовый файл, пробуя кодировки по порядку."""
    for enc in ENCODINGS:
        try:
            return path.read_text(encoding=enc).splitlines()
        except (UnicodeDecodeError, UnicodeError):
            continue
    # latin-1 не падает при декодировании — надёжный запасной вариант
    return path.read_text(encoding="latin-1", errors="replace").splitlines()


def _numbers_in_line(line: str) -> List[float]:
    """Все числа в строке (по порядку), float; пустой список, если чисел нет."""
    vals = []
    for m in NUMBER_RE.finditer(line):
        try:
            vals.append(float(m.group(0)))
        except ValueError:
            continue
    return vals


def _parse_column(path: Path, start_line: int, log: Optional[list], label: str) -> List[float]:
    """Разобрать одноколоночный .dat-файл (значения x или T).

    Числа читаются по порядку из каждой строки, начиная со start_line.
    Пустые строки, комментарии (#, %, //) и нечисловые заголовки игнорируются;
    длинные логические строки, перенесённые на несколько физических строк
    (как в выгрузках GraphLab), собираются корректно.
    Нефинитные значения (NaN/inf) пропускаются с предупреждением.
    """
    values: List[float] = []
    headers_skipped = 0
    lines = _read_lines(path, log)
    for idx, raw in enumerate(lines, start=1):
        if idx < start_line:
            continue  # строка до начала данных
        line = raw.strip()
        if not line:
            continue  # пустая строка
        if line.startswith(("#", "%", "//")):
            continue  # комментарий
        low = line.lower()
        if low in {"nan", "inf", "-inf", "+inf", "infinity"}:
            _log(log, LVL_WARN, f"{label}: строка {idx} — нефинитное значение ({line!r}), пропущено")
            continue
        vals = _numbers_in_line(line)
        if not vals:
            headers_skipped += 1  # заголовок / нечисловая строка
            continue
        for v in vals:
            if math.isfinite(v):
                values.append(v)
            else:
                _log(log, LVL_WARN, f"{label}: строка {idx} — нефинитное значение ({v!r}), пропущено")
    if headers_skipped:
        _log(log, LVL_INFO, f"{label}: пропущено нечисловых строк (заголовки): {headers_skipped}")
    return values


def parse_separate_files(x_path, t_path, start_line_x=1, start_line_t=1, log=None) -> List[Tuple[float, float]]:
    """Прочитать два раздельных .dat-файла (x и T) и собрать пары (x, T).

    В каждом файле ожидается одно число на строку; если в строке несколько
    значений, все берутся по порядку (перенос длинных строк GraphLab).
    Если количество значений не совпадает, используется минимальное
    количество пар (с предупреждением в логе).
    Бросает FileNotFoundError, если файл не найден.
    """
    x_path = Path(x_path)
    t_path = Path(t_path)
    for p, name in ((x_path, "файл с x"), (t_path, "файл с T")):
        if not p.exists():
            raise FileNotFoundError(f"{name} не найден: {p}")
    xs = _parse_column(x_path, start_line_x, log, "файл x")
    ts = _parse_column(t_path, start_line_t, log, "файл T")
    _log(log, LVL_INFO, f"прочитано значений: x — {len(xs)}, T — {len(ts)}")
    n = min(len(xs), len(ts))
    if len(xs) != len(ts):
        _log(log, LVL_WARN,
             f"несовпадение количества: x — {len(xs)}, T — {len(ts)}; используется N = {n} пар")
    return list(zip(xs[:n], ts[:n]))


# ------------------------------ расчёты ------------------------------

def integrate_theta(x, T, L, T0, T1):
    """Интеграл I = ∫[0..L] (T(x) − T1) dx: трапеции + краевые заплатки.

    x, T — отсортированные по возрастанию x конечные пары в пределах [0, L];
    T0, T1 — в кельвинах. Возвращает (I, детали).
    """
    n = len(x)
    if n < 2:
        raise ValueError("нужно минимум 2 точки данных")
    theta = [t - T1 for t in T]  # T(x) − T1, К

    # основная часть: метод трапеций на [x_min, x_max]
    I_main = 0.0
    for i in range(n - 1):
        I_main += (x[i + 1] - x[i]) * (theta[i] + theta[i + 1]) / 2.0

    patches: List[Tuple[str, float]] = []
    I = I_main
    # левая заплатка: [0, x_min], T(0) = T0 (заданное основание)
    if x[0] > 0.0:
        p0 = x[0] * ((T0 - T1) + theta[0]) / 2.0
        I += p0
        patches.append((f"[0, {x[0]:.4g}] с T(0)=T0", p0))
    # правая заплатка: [x_max, L], T(L) = T(x_max) (адиабатический торец)
    if x[-1] < L:
        pL = (L - x[-1]) * theta[-1]
        I += pL
        patches.append((f"[{x[-1]:.4g}, {L:.4g}] с T(L)=T(x_max)", pL))

    details = {
        "n": n,
        "I_main": I_main,
        "patches": patches,
        "T_L": T[-1],      # оценка T(L) — последняя точка профиля
        "x_min": x[0],
        "x_max": x[-1],
    }
    return I, details


def _ssr_for_M(M: float, x, theta, L: float) -> float:
    """Сумма квадратов остатков для фиксированного M (A — по НМК)."""
    s_g2 = 0.0
    s_gt = 0.0
    gs = []
    for xi, ti in zip(x, theta):
        g = math.cosh(M * (1.0 - xi / L))
        gs.append(g)
        s_g2 += g * g
        s_gt += g * ti
    if s_g2 == 0.0:
        return float("inf")
    A = s_gt / s_g2
    ssr = 0.0
    for g, ti in zip(gs, theta):
        r = ti - A * g
        ssr += r * r
    return ssr


def fit_M(x, theta, L) -> Tuple[float, float]:
    """Подгонка M из θ(x) = A·cosh[M·(1 − x/L)].

    A для фиксированного M — линейно по наименьшим квадратам:
    A = Σ(θ·g)/Σ(g²). M минимизирует сумму квадратов остатков методом
    золотого сечения на [0.05, 20] до ширины интервала ~1e-10.
    Возвращает (M, относительный остаток), где
    относительный остаток = sqrt(SSR / Σθ²).
    """
    n = len(x)
    if n < 2:
        raise ValueError("нужно минимум 2 точки данных")
    s_t2 = sum(t * t for t in theta)

    a, b = M_FIT_LO, M_FIT_HI
    inv_phi = (math.sqrt(5.0) - 1.0) / 2.0  # 1/φ ≈ 0.618
    c = b - inv_phi * (b - a)
    d = a + inv_phi * (b - a)
    fc = _ssr_for_M(c, x, theta, L)
    fd = _ssr_for_M(d, x, theta, L)
    while (b - a) > M_FIT_TOL:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - inv_phi * (b - a)
            fc = _ssr_for_M(c, x, theta, L)
        else:
            a, c, fc = c, d, fd
            d = a + inv_phi * (b - a)
            fd = _ssr_for_M(d, x, theta, L)
    M = (a + b) / 2.0
    ssr = _ssr_for_M(M, x, theta, L)
    rel_res = 0.0 if s_t2 <= 1e-30 else math.sqrt(ssr / s_t2)
    return M, rel_res


def compute_results(b, L, alpha, t0_c, t1_c, x_path, t_path, start_x, start_t, log=None):
    """Полный расчёт по параметрам и .dat-файлам.

    Возвращает (params, results) — словари для отображения и отчёта.
    Бросает ValueError / FileNotFoundError с понятными сообщениями.
    """
    # --- валидация параметров ---
    for name, v in (("b", b), ("L", L), ("α", alpha)):
        if not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError(f"{name}: введите конечное число")
        if v <= 0:
            raise ValueError(f"{name} должен быть > 0 (введено: {v})")
    for name, v in (("T0", t0_c), ("T1", t1_c)):
        if not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError(f"{name}: введите конечное число (°C)")

    t0 = t0_c + KELVIN_OFFSET  # К
    t1 = t1_c + KELVIN_OFFSET  # К

    # --- пути к файлам ---
    x_path = Path(str(x_path).strip())
    t_path = Path(str(t_path).strip())
    if not str(x_path):
        raise ValueError("не выбран файл с x")
    if not str(t_path):
        raise ValueError("не выбран файл с T")

    # --- чтение данных ---
    pairs = parse_separate_files(x_path, t_path, start_x, start_t, log)
    if len(pairs) < 2:
        raise ValueError("меньше двух точек данных во входных файлах")

    # сортировка по возрастанию x
    pairs.sort(key=lambda p: p[0])
    # исключение точек вне [0, L] больше чем на tol = 1e-9·L
    tol = RANGE_TOL_FACTOR * L
    kept = [(x, t) for (x, t) in pairs if -tol <= x <= L + tol]
    dropped = len(pairs) - len(kept)
    if dropped:
        _log(log, LVL_WARN, f"точек вне диапазона [0, L], исключено из расчёта: {dropped}")
    if len(kept) < 2:
        raise ValueError("меньше двух точек данных в диапазоне [0, L]")
    xs = [p[0] for p in kept]
    ts = [p[1] for p in kept]

    _log(log, LVL_INFO, f"пар (x, T) прочитано и использовано: {len(kept)}")

    # --- интеграл I ---
    I, det = integrate_theta(xs, ts, L, t0, t1)
    P = 2.0 * b
    A_f = 2.0 * b * L
    for name, contrib in det["patches"]:
        _log(log, LVL_INFO, f"краевая заплатка {name}: вклад {contrib:.4g} К·м")

    dt = t0 - t1
    if abs(dt) < TOL_DT_EQ:
        # особый случай: разности температур нет — не делим на ноль
        _log(log, LVL_WARN, "|T0 − T1| < 1e-12: Q̇ = 0, η_f не определена (показывается «—»)")
        Q = 0.0
        eta_f = None
        M = rel_res = eta_an = disc = None
    else:
        Q = alpha * P * I
        eta_f = I / (L * dt)
        # sanity: первая точка профиля против температуры основания
        if abs(ts[0] - t0) > 1.0:
            _log(log, LVL_WARN,
                 f"первая точка профиля отличается от T0 больше чем на 1 К "
                 f"({ts[0]:.4g} К против {t0:.4g} К)")
        # нормированный профиль θ и подгонка M
        theta = [(t - t1) / dt for t in ts]
        M, rel_res = fit_M(xs, theta, L)
        eta_an = math.tanh(M) / M
        disc = abs(eta_f - eta_an) / eta_an * 100.0
        if rel_res > FIT_WARN_REL_RES:
            _log(log, LVL_WARN,
                 f"плохая подгонка M: относительный остаток {rel_res * 100:.2f} % > 5 %")

    params = {
        "b": b, "L": L, "alpha": alpha,
        "t0_c": t0_c, "t1_c": t1_c, "t0_k": t0, "t1_k": t1,
        "x_path": str(x_path), "t_path": str(t_path),
        "start_x": start_x, "start_t": start_t,
    }
    results = {
        "P": P, "A_f": A_f, "I": I, "Q": Q, "eta_f": eta_f,
        "M": M, "rel_res": rel_res, "eta_analytic": eta_an, "disc_pct": disc,
        "T_L": det["T_L"], "n_pairs": len(kept),
        "x_min": xs[0], "x_max": xs[-1],
        "patches": det["patches"],
    }
    return params, results


# ------------------------------ отчёт ------------------------------

def write_report(path, params, results, log_lines):
    """Записать текстовый отчёт .txt (UTF-8). Возвращает путь."""
    path = Path(path)
    out: List[str] = []
    add = out.append
    bar = "=" * 64
    add(bar)
    add("Отчёт: плоское ребро — мощность теплоотдачи Q̇ и эффективность η_f")
    add(f"Дата/время: {datetime.now():%Y-%m-%d %H:%M:%S}")
    add(bar)
    add("")
    add("Входные параметры")
    add("-----------------")
    add(f"b  = {fmt4(params['b'])} м")
    add(f"L  = {fmt4(params['L'])} м")
    add(f"α  = {fmt4(params['alpha'])} Вт/(м²·К)")
    add(f"T0 = {fmt4(params['t0_c'])} °C = {params['t0_k']:.2f} К")
    add(f"T1 = {fmt4(params['t1_c'])} °C = {params['t1_k']:.2f} К")
    add("")
    add("Входные данные")
    add("---------------")
    add(f"файл x: {params['x_path']} (строка начала: {params['start_x']})")
    add(f"файл T: {params['t_path']} (строка начала: {params['start_t']})")
    add(f"пар использовано: {results['n_pairs']}")
    add(f"диапазон x: [{fmt4(results['x_min'])}, {fmt4(results['x_max'])}] м")
    add("")
    add("Результаты")
    add("----------")
    add(f"P = 2b             = {fmt4(results['P'])} м")
    add(f"A_f = 2bL          = {fmt4(results['A_f'])} м²")
    add(f"I                  = {fmt4(results['I'])} К·м")
    add(f"Q̇                 = {fmt4(results['Q'])} Вт")
    add(f"η_f                = {fmt4(results['eta_f'])}")
    add(f"M (подгонка)       = {fmt4(results['M'])}")
    add(f"η_analytic         = {fmt4(results['eta_analytic'])}   (= tanh(M)/M)")
    add(f"расхождение        = {fmt4(results['disc_pct'])} %")
    add(f"T(L) (оценка)      = {fmt4(results['T_L'])} К")
    if results["patches"]:
        add("")
        add("Краевые заплатки интеграла:")
        for name, contrib in results["patches"]:
            add(f"  - {name}: {contrib:.4g} К·м")
    add("")
    add("Предупреждения")
    add("--------------")
    warns = [ln.split(": ", 1)[1] for ln in (log_lines or [])
             if ln.startswith((LVL_WARN + ":", LVL_ERROR + ":"))]
    if warns:
        for w in warns:
            add(f"- {w}")
    else:
        add("- предупреждений нет")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return path


# ------------------------------ GUI (tkinter) ------------------------------

class FinCalculatorApp:
    """Графический интерфейс приложения (tkinter)."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("Плоское ребро — расчёт Q̇ и η_f")
        root.geometry("780x840")
        root.minsize(680, 640)

        self.log_lines: List[str] = []
        self.last_params: Optional[dict] = None
        self.last_results: Optional[dict] = None

        self._build_ui()
        self._update_k_labels()

    # ---------- построение UI ----------

    def _build_ui(self) -> None:
        outer = tk.Frame(self.root)
        outer.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        outer.columnconfigure(0, weight=1)

        self._build_params_frame(outer)
        self._build_data_frame(outer)
        self._build_buttons_frame(outer)
        self._build_results_frame(outer)
        self._build_log_frame(outer)

    def _build_params_frame(self, parent) -> None:
        frm = tk.LabelFrame(parent, text="Параметры расчёта")
        frm.grid(row=0, column=0, sticky="ew")
        frm.columnconfigure(1, weight=1)

        self.var_b = tk.StringVar(value=f"{DEFAULT_B:.3f}")
        self.var_L = tk.StringVar(value=f"{DEFAULT_L:.3f}")
        self.var_alpha = tk.StringVar(value=f"{DEFAULT_ALPHA:.1f}")
        self.var_t0 = tk.StringVar(value=f"{DEFAULT_T0_C:.1f}")
        self.var_t1 = tk.StringVar(value=f"{DEFAULT_T1_C:.1f}")

        r = 0
        for text, var in (
            ("b — ширина ребра, м", self.var_b),
            ("L — высота охлаждаемой части, м", self.var_L),
            ("α — коэффициент теплоотдачи, Вт/(м²·К)", self.var_alpha),
        ):
            tk.Label(frm, text=text).grid(row=r, column=0, sticky="w", padx=(0, 8), pady=2)
            tk.Entry(frm, textvariable=var, width=12).grid(row=r, column=1, sticky="w", pady=2)
            r += 1

        tk.Label(frm, text="T0 — температура основания, °C").grid(
            row=r, column=0, sticky="w", padx=(0, 8), pady=2)
        e0 = tk.Entry(frm, textvariable=self.var_t0, width=12)
        e0.grid(row=r, column=1, sticky="w", pady=2)
        self.lbl_t0_k = tk.Label(frm, text="", fg="#0a58ca")
        self.lbl_t0_k.grid(row=r, column=2, sticky="w", padx=(12, 0), pady=2)
        e0.bind("<KeyRelease>", lambda _e: self._update_k_labels())
        r += 1

        tk.Label(frm, text="T1 — температура воздуха, °C").grid(
            row=r, column=0, sticky="w", padx=(0, 8), pady=2)
        e1 = tk.Entry(frm, textvariable=self.var_t1, width=12)
        e1.grid(row=r, column=1, sticky="w", pady=2)
        self.lbl_t1_k = tk.Label(frm, text="", fg="#0a58ca")
        self.lbl_t1_k.grid(row=r, column=2, sticky="w", padx=(12, 0), pady=2)
        e1.bind("<KeyRelease>", lambda _e: self._update_k_labels())
        r += 1

        tk.Label(frm, text="η_f от α не зависит (α сокращается в формуле), "
                           "с α меняется только Q̇.", fg="#555555").grid(
            row=r, column=0, columnspan=3, sticky="w", pady=(6, 0))

    def _build_data_frame(self, parent) -> None:
        frm = tk.LabelFrame(parent, text="Входные данные (.dat-файлы)")
        frm.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        frm.columnconfigure(2, weight=1)

        self.var_x_path = tk.StringVar(value="")
        self.var_t_path = tk.StringVar(value="")
        self.var_start_x = tk.StringVar(value="1")
        self.var_start_t = tk.StringVar(value="4")

        # строка: файл с x
        tk.Label(frm, text="Файл с x…").grid(row=0, column=0, sticky="w", pady=2)
        tk.Button(frm, text="выбрать", width=10,
                  command=lambda: self._choose_file(self.var_x_path)).grid(
            row=0, column=1, padx=(4, 4), pady=2)
        tk.Entry(frm, textvariable=self.var_x_path).grid(
            row=0, column=2, sticky="ew", pady=2)
        tk.Label(frm, text="строка начала:").grid(row=0, column=3, sticky="e", padx=(8, 2), pady=2)
        tk.Entry(frm, textvariable=self.var_start_x, width=5).grid(row=0, column=4, sticky="w", pady=2)

        # строка: файл с T
        tk.Label(frm, text="Файл с T…").grid(row=1, column=0, sticky="w", pady=2)
        tk.Button(frm, text="выбрать", width=10,
                  command=lambda: self._choose_file(self.var_t_path)).grid(
            row=1, column=1, padx=(4, 4), pady=2)
        tk.Entry(frm, textvariable=self.var_t_path).grid(
            row=1, column=2, sticky="ew", pady=2)
        tk.Label(frm, text="строка начала:").grid(row=1, column=3, sticky="e", padx=(8, 2), pady=2)
        tk.Entry(frm, textvariable=self.var_start_t, width=5).grid(row=1, column=4, sticky="w", pady=2)

        tk.Label(frm, text="(для T по умолчанию 4 — рекомендуется для GraphLab: "
                           "заголовок + нулевой ряд)", fg="#555555").grid(
            row=2, column=0, columnspan=5, sticky="w", pady=(4, 0))

    def _build_buttons_frame(self, parent) -> None:
        frm = tk.Frame(parent)
        frm.grid(row=2, column=0, sticky="ew", pady=8)
        tk.Button(frm, text="Рассчитать", width=16, command=self.calculate).pack(
            side=tk.LEFT, padx=(0, 8))
        tk.Button(frm, text="Сохранить отчёт…", width=16, command=self.save_report).pack(
            side=tk.LEFT)

    def _build_results_frame(self, parent) -> None:
        frm = tk.LabelFrame(parent, text="Результаты")
        frm.grid(row=3, column=0, sticky="ew")
        self.txt_results = tk.Text(frm, height=10, state=tk.DISABLED,
                                   background="#f7f7f2", relief=tk.FLAT)
        self.txt_results.pack(fill=tk.BOTH, expand=True)

    def _build_log_frame(self, parent) -> None:
        frm = tk.LabelFrame(parent, text="Лог")
        frm.grid(row=4, column=0, sticky="ew", pady=(8, 0))
        inner = tk.Frame(frm)
        inner.pack(fill=tk.BOTH, expand=True)
        self.txt_log = tk.Text(inner, height=10, relief=tk.FLAT)
        sb = tk.Scrollbar(inner, command=self.txt_log.yview)
        self.txt_log.configure(yscrollcommand=sb.set)
        self.txt_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

    # ---------- обработчики ----------

    def _update_k_labels(self) -> None:
        """Live-метки Kельвины рядом с T0 и T1 (при некорректном вводе — «—»)."""
        for var, lbl in ((self.var_t0, self.lbl_t0_k), (self.var_t1, self.lbl_t1_k)):
            v = self._parse_entry(var)
            lbl.config(text=f"{v + KELVIN_OFFSET:.2f} К" if v is not None else "—")

    @staticmethod
    def _parse_entry(var) -> Optional[float]:
        """Число из поля ввода (запятая → точка); None, если некорректно."""
        s = var.get().strip().replace(",", ".")
        if not s:
            return None
        try:
            v = float(s)
        except ValueError:
            return None
        return v if math.isfinite(v) else None

    @staticmethod
    def _parse_start_line(var, label) -> int:
        """Целая стартовая строка данных (≥ 1)."""
        s = var.get().strip()
        try:
            v = int(s)
        except ValueError:
            raise ValueError(f"строка начала для файла {label}: введите целое число (≥ 1)")
        if v < 1:
            raise ValueError(f"строка начала для файла {label} должна быть ≥ 1")
        return v

    def _choose_file(self, var: tk.StringVar) -> None:
        """Диалог выбора .dat-файла."""
        path = filedialog.askopenfilename(
            title="Выбрать .dat-файл",
            filetypes=[("DAT-файлы (*.dat)", "*.dat"), ("Все файлы (*.*)", "*.*")])
        if path:
            var.set(path)

    def calculate(self) -> None:
        """Кнопка «Рассчитать»: параметры + файлы → результаты и лог."""
        self.log_lines = []
        log = self.log_lines
        try:
            b = self._parse_entry(self.var_b)
            L = self._parse_entry(self.var_L)
            alpha = self._parse_entry(self.var_alpha)
            t0_c = self._parse_entry(self.var_t0)
            t1_c = self._parse_entry(self.var_t1)
            for name, v in (("b", b), ("L", L), ("α", alpha), ("T0", t0_c), ("T1", t1_c)):
                if v is None:
                    raise ValueError(f"{name}: введите конечное число")

            start_x = self._parse_start_line(self.var_start_x, "x")
            start_t = self._parse_start_line(self.var_start_t, "T")

            params, results = compute_results(
                b, L, alpha, t0_c, t1_c,
                self.var_x_path.get().strip(), self.var_t_path.get().strip(),
                start_x, start_t, log)
        except Exception as e:  # никаких необработанных исключений
            _log(log, LVL_ERROR, str(e))
            self._show_log()
            messagebox.showerror("Ошибка расчёта", str(e))
            return

        self.last_params = params
        self.last_results = results
        self._show_results(results)
        self._show_log()

    def save_report(self) -> None:
        """Кнопка «Сохранить отчёт…»: запись .txt (UTF-8)."""
        if self.last_results is None or self.last_params is None:
            messagebox.showinfo("Отчёт", "Нет результатов — сначала нажмите «Рассчитать».")
            return
        path = filedialog.asksaveasfilename(
            title="Сохранить отчёт",
            defaultextension=".txt",
            initialfile="fin_report.txt",
            filetypes=[("Текстовые файлы (*.txt)", "*.txt"), ("Все файлы (*.*)", "*.*")])
        if not path:
            return
        try:
            write_report(path, self.last_params, self.last_results, self.log_lines)
        except Exception as e:
            messagebox.showerror("Ошибка записи отчёта", str(e))
            return
        messagebox.showinfo("Отчёт", f"Отчёт сохранён:\n{path}")

    # ---------- отображение ----------

    def _show_results(self, results) -> None:
        lines = [
            f"P = 2b                  : {fmt4(results['P'])} м",
            f"A_f = 2bL               : {fmt4(results['A_f'])} м²",
            f"I = ∫(T−T1)dx           : {fmt4(results['I'])} К·м",
            f"Q̇ = α·P·I               : {fmt4(results['Q'])} Вт",
            f"η_f                     : {fmt4(results['eta_f'])}",
            f"M (подгонка)            : {fmt4(results['M'])}",
            f"η_analytic = tanh(M)/M  : {fmt4(results['eta_analytic'])}",
            f"расхождение             : {fmt4(results['disc_pct'])} %",
            f"T(L) (оценка, торец)    : {fmt4(results['T_L'])} К",
        ]
        self._set_text(self.txt_results, "\n".join(lines), readonly=True)

    def _show_log(self) -> None:
        text = "\n".join(self.log_lines) if self.log_lines else "(записей нет)"
        self._set_text(self.txt_log, text, readonly=False)

    @staticmethod
    def _set_text(widget: tk.Text, text: str, readonly: bool = True) -> None:
        widget.configure(state=tk.NORMAL)
        widget.delete("1.0", tk.END)
        widget.insert(tk.END, text)
        widget.configure(state=tk.DISABLED if readonly else tk.NORMAL)
        widget.see(tk.END)


# ------------------------------ точка входа ------------------------------

def main() -> None:
    """Запуск графического интерфейса."""
    root = tk.Tk()
    FinCalculatorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
