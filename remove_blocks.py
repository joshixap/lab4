import pandas as pd
import numpy as np

def remove_blocks_2d(df, percent, block_shape_range=((2,2), (4,4)), seed=42):
    """
    Удаляет (устанавливает NaN) ячейки в DataFrame df исключительно БЛОКАМИ указанного размерного диапазона.
    percent: процент от общего числа ячеек, который нужно "удалить" (заполнить NaN).
    block_shape_range: ((min_rows, min_cols), (max_rows, max_cols)) — только диапазон, без одиночного режима!
    seed: фиксирует random seed для воспроизводимости.
    """
    np.random.seed(seed)
    df = df.copy()
    n_rows, n_cols = df.shape
    total_cells = n_rows * n_cols

    # Только диапазон блоков!
    if (isinstance(block_shape_range, tuple) and len(block_shape_range) == 2
        and isinstance(block_shape_range[0], tuple) and isinstance(block_shape_range[1], tuple)):
        min_block_rows, min_block_cols = block_shape_range[0]
        max_block_rows, max_block_cols = block_shape_range[1]
    else:
        # Ошибка, если не диапазон: файл должен быть вызван только с диапазоном!
        raise ValueError("block_shape_range должен быть ТОЛЬКО диапазоном ((min_r,min_c),(max_r,max_c))")

    est_block_rows = (min_block_rows + max_block_rows) // 2
    est_block_cols = (min_block_cols + max_block_cols) // 2
    block_size = est_block_rows * est_block_cols
    required_nans = int(total_cells * percent / 100)

    nan_count = 0
    # Флаг для ограничения по прогрессу
    tries = 0
    max_tries = 10000  # чтобы не зависнуть навечно
    while nan_count < required_nans and tries < max_tries:
        tries += 1
        br = np.random.randint(min_block_rows, max_block_rows + 1)
        bc = np.random.randint(min_block_cols, max_block_cols + 1)
        # если блок больше размера DataFrame — перескочить
        if n_rows - br + 1 <= 0 or n_cols - bc + 1 <= 0:
            continue
        row_start = np.random.randint(0, n_rows - br + 1)
        col_start = np.random.randint(0, n_cols - bc + 1)
        # Найти незаполненные ячейки в этом блоке
        block_indices = [(r, c) for r in range(row_start, row_start + br) for c in range(col_start, col_start + bc)]
        added = 0
        for r, c in block_indices:
            # Только незаполненные превращать в NaN!
            if not pd.isnull(df.iat[r, c]):
                df.iat[r, c] = np.nan
                nan_count += 1
                added += 1
                if nan_count >= required_nans:
                    break
        if nan_count >= required_nans:
            break

    # В конце: если не удалось набрать процент — просто закончить (не одиночными, только блоками!)
    # Можно добавить print(f"Итого поставлено NaN: {nan_count} из {required_nans}")

    return df

if __name__ == "__main__":
    input_file = "isodata.csv"
    output_file = "isodata_blocks.csv"

    percent = 5
    block_shape_range = ((2,2), (4,4)) # диапазон БЛОКОВ

    df = pd.read_csv(input_file, sep=",")
    df_na = remove_blocks_2d(df, percent=percent, block_shape_range=block_shape_range)
    df_na.to_csv(output_file, index=False)