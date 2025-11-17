import pandas as pd
import numpy as np

def remove_blocks_2d(df, percent, block_shape_range=((2,2), (4,4)), seed=42):
    """
    Удаляет (устанавливает NaN) ячейки в DataFrame df блоками заданного размера.
    percent: процент от общего числа ячеек, который нужно "удалить" (заполнить NaN).
    block_shape_range: ((min_rows, min_cols), (max_rows, max_cols)), либо (rows, cols) для фиксированного размера блока.
    seed: фиксирует random seed для воспроизводимости.
    """
    np.random.seed(seed)
    df = df.copy()
    n_rows, n_cols = df.shape
    total_cells = n_rows * n_cols

    # Исправленная логика диапазона/фиксированного блока
    if (isinstance(block_shape_range, tuple) and len(block_shape_range) == 2
        and all(isinstance(x, int) for x in block_shape_range)):
        min_block_rows = max_block_rows = block_shape_range[0]
        min_block_cols = max_block_cols = block_shape_range[1]
    elif (isinstance(block_shape_range, tuple) and len(block_shape_range) == 2
          and isinstance(block_shape_range[0], tuple) and isinstance(block_shape_range[1], tuple)):
        min_block_rows, min_block_cols = block_shape_range[0]
        max_block_rows, max_block_cols = block_shape_range[1]
    else:
        # По умолчанию диапазон ((2,2), (4,4))
        min_block_rows, min_block_cols = (2, 2)
        max_block_rows, max_block_cols = (4, 4)

    est_block_rows = (min_block_rows + max_block_rows) // 2
    est_block_cols = (min_block_cols + max_block_cols) // 2
    block_size = est_block_rows * est_block_cols    
    n_blocks = max(int(np.ceil((total_cells * percent / 100) / block_size)), 2)

    for _ in range(n_blocks):
        br = np.random.randint(min_block_rows, max_block_rows + 1)
        bc = np.random.randint(min_block_cols, max_block_cols + 1)
        # Следим, чтобы блок не выходил за границы
        if n_rows - br + 1 <= 0 or n_cols - bc + 1 <= 0:
            continue
        row_start = np.random.randint(0, n_rows - br + 1)
        col_start = np.random.randint(0, n_cols - bc + 1)
        for r in range(row_start, row_start + br):
            for c in range(col_start, col_start + bc):
                df.iat[r, c] = np.nan
    return df

if __name__ == "__main__":
    # Просто запускает с дефолтными параметрами на isodata.csv,
    # сохраняет результат в isodata_blocks.csv
    input_file = "isodata.csv"
    output_file = "isodata_blocks.csv"

    percent = 5  # процент пропусков
    block_shape_range = ((2,2), (4,4))  # диапазон блока, КАК В АРГУМЕНТЕ ФУНКЦИИ

    df = pd.read_csv(input_file, sep=",")
    df_na = remove_blocks_2d(df, percent=percent, block_shape_range=block_shape_range)
    df_na.to_csv(output_file, index=False)