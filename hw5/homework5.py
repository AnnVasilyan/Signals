import numpy as np
from itertools import product

# ----------------------------------------------------------------------
# вспомогательные функции для работы с двоичными векторами
# ----------------------------------------------------------------------
def _to_binary_vector(number, bits_count):
    """представляет целое число в виде двоичного вектора длины bits_count (старший бит слева)"""
    # возвращаем список битов, number > 0
    binary = []
    temp = number
    for _ in range(bits_count):
        binary.append(temp & 1)   # младший бит
        temp >>= 1
    return binary[::-1]            # переворачиваем, чтобы старший был первым

def _weight(vector):
    """вес Хэмминга (число единиц)"""
    return int(np.sum(vector))

def _distance(vec_a, vec_b):
    """расстояние Хэмминга между двумя двоичными векторами"""
    return int(np.sum(np.bitwise_xor(vec_a, vec_b)))

def _all_code_vectors(generator):
    """перебор всех кодовых слов линейного кода с порождающей матрицей generator (k x n)"""
    k, n = generator.shape
    messages = product([0, 1], repeat=k)
    result = []
    for msg in messages:
        msg_arr = np.array(msg, dtype=int)
        code_vec = (msg_arr @ generator) % 2
        result.append((msg_arr, code_vec))
    return result

# ----------------------------------------------------------------------
# построение матрицы Хэмминга
# ----------------------------------------------------------------------
def _hamming_parity_matrix(r):
    """
    Строит проверочную матрицу H для двоичного кода Хэмминга с параметром r.
    Возвращает матрицу размера r x (2^r - 1), столбцы – все ненулевые r-битные векторы.
    """
    n = (1 << r) - 1               # 2**r - 1
    H = np.zeros((r, n), dtype=int)

    for col_idx in range(n):
        number = col_idx + 1        # числа от 1 до 2^r-1
        bits = _to_binary_vector(number, r)
        H[:, col_idx] = bits
    return H

# ----------------------------------------------------------------------
# основная проверка свойства симплексности
# ----------------------------------------------------------------------
def examine_simplex_property(r):
    """
    Для заданного r строит дуальный код к коду Хэмминга (порождающая матрица = H)
    и проверяет, является ли он симплексным:
    - все ненулевые кодовые слова имеют одинаковый вес 2^(r-1)
    - расстояние между любыми двумя различными словами тоже равно 2^(r-1)
    """
    print("=" * 72)
    print(f"  Параметр r = {r}")
    print(f"  Код Хэмминга: (n={2**r-1}, k={2**r-1-r}), d_min=3")
    print(f"  Дуальный код: (n={2**r-1}, k={r})")
    print(f"  Ожидаемый вес ненулевых слов: 2^(r-1) = {2**(r-1)}")
    print("=" * 72)

    # 1) строим матрицу H
    H_matrix = _hamming_parity_matrix(r)
    n_cols = H_matrix.shape[1]       # = 2**r - 1
    k_dual = H_matrix.shape[0]       # = r

    print("\n--- Проверочная матрица Хэмминга (она же порождающая для дуального) ---")
    for i in range(k_dual):
        row_str = " ".join(map(str, H_matrix[i]))
        print(f"  строка {i+1}: [{row_str}]")

    # 2) генерируем все слова дуального кода
    dual_code_vectors = _all_code_vectors(H_matrix)   # список пар (инф.вектор, кодовое слово)
    total_words = len(dual_code_vectors)
    print(f"\nВсего кодовых слов: {total_words} = 2^{k_dual}\n")

    # 3) собираем веса и расстояния
    weight_values = []
    nonzero_weights = []

    # для красивого вывода заголовка
    print(f"{'инф.слово':>12s}  |  {'кодовое слово':>27s}  |  вес")
    print("-" * 62)

    for msg, code in dual_code_vectors:
        w = _weight(code)
        weight_values.append(w)
        if w > 0:
            nonzero_weights.append(w)

        msg_str = ''.join(str(b) for b in msg)
        code_str = ''.join(str(b) for b in code)
        print(f"  {msg_str:>10s}  |  {code_str:>27s}  |  {w}")

    # 4) анализ весов
    expected_weight = 2**(r-1)
    unique_nonzero_weights = set(nonzero_weights)
    all_nonzero_equal = (len(unique_nonzero_weights) == 1) and (expected_weight in unique_nonzero_weights)

    print("\n--- Анализ весов ---")
    print(f"  ожидаемый вес ненулевых слов: {expected_weight}")
    print(f"  реальные веса (уникальные): {sorted(unique_nonzero_weights)}")
    print(f"  все ненулевые веса одинаковы: {'да' if all_nonzero_equal else 'нет'}")

    # 5) анализ расстояний между различными словами
    distinct_distances = set()
    for i in range(total_words):
        for j in range(i+1, total_words):
            d = _distance(dual_code_vectors[i][1], dual_code_vectors[j][1])
            if d > 0:
                distinct_distances.add(d)

    print("\n--- Анализ расстояний ---")
    print(f"  множество расстояний между разными словами: {sorted(distinct_distances)}")
    all_distances_equal = (len(distinct_distances) == 1)
    print(f"  все расстояния одинаковы: {'да' if all_distances_equal else 'нет'}")

    # 6) полное распределение весов
    from collections import Counter
    freq = Counter(weight_values)
    print("\n--- Распределение весов ---")
    for w in sorted(freq):
        print(f"  вес {w}: {freq[w]} слов")

    # 7) итоговый вывод
    is_simplex = all_nonzero_equal and all_distances_equal
    print(f"\n{'=' * 72}")
    if is_simplex:
        print(f"  РЕЗУЛЬТАТ: дуальный код к коду Хэмминга (r={r}) ЯВЛЯЕТСЯ симплексным.")
        print(f"  Все {len(nonzero_weights)} ненулевых слов имеют вес {expected_weight}.")
        print(f"  Расстояние между любыми двумя различными словами = {expected_weight}.")
    else:
        print(f"  РЕЗУЛЬТАТ: код НЕ является симплексным.")
    print(f"{'=' * 72}\n")

    return is_simplex

# ----------------------------------------------------------------------
# точка входа: запускаем проверку для нескольких значений r
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("ПРОВЕРКА ГИПОТЕЗЫ: дуальный код к коду Хэмминга – симплексный\n")

    test_parameters = [2, 3, 4, 5]      # можно расширить до 6,7, но для наглядности ограничимся
    outcomes = {}

    for param in test_parameters:
        outcomes[param] = examine_simplex_property(param)

    # сводная таблица
    print("\n" + "=" * 72)
    print("  СВОДНАЯ ТАБЛИЦА")
    print("=" * 72)
    header = f"  {'r':>3s}  {'(n, k)':>10s}  {'ожид.вес':>8s}  {'факт.вес':>8s}  {'симплекс?':>9s}"
    print(header)
    print("-" * 56)

    for r in test_parameters:
        n_val = (1 << r) - 1
        k_val = r
        exp_weight = 1 << (r-1)          # 2^(r-1)
        # фактический вес берём из предыдущих вычислений (для отчёта используем ожидаемый, если код симплексный)
        is_ok = outcomes[r]
        fact = str(exp_weight) if is_ok else "разный"
        yesno = "ДА" if is_ok else "НЕТ"
        print(f"  {r:>3d}  ({n_val:>2d}, {k_val:>2d})  {exp_weight:>8d}  {fact:>8s}  {yesno:>9s}")

    print()