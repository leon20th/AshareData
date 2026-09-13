import io
import os

import pandas as pd


def read_csv_by_index(file_path, start_index=None, end_index=None, index_dtype='str', prepend_n_rows=0):
    """Read rows whose first-column index in the inclusive range [start_index, end_index].

    If both bounds are None, read the whole file. If one bound is None, use a one-sided inclusive range.
    When start_index is provided, prepend_n_rows adds up to N rows immediately before the matched start row.
    """
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        raise FileNotFoundError(f"CSV file not found or empty: {file_path}")

    prepend_n_rows = int(prepend_n_rows)
    if prepend_n_rows < 0:
        raise ValueError("prepend_n_rows must be greater than or equal to 0")

    if start_index is None and end_index is None:
        return pd.read_csv(file_path)

    def normalize_index(value):
        if value is None:
            return None
        if index_dtype == 'int':
            return int(value)
        if index_dtype == 'float':
            return float(value)
        return str(value)

    def read_row_at_or_after(csv_file, offset, file_size, header_end):
        if offset <= header_end:
            csv_file.seek(header_end)
        else:
            csv_file.seek(offset - 1)
            prev_byte = csv_file.read(1)
            if prev_byte == b'\n':
                csv_file.seek(offset)
            else:
                csv_file.seek(offset)
                csv_file.readline()
        row_offset = csv_file.tell()
        if row_offset >= file_size:
            return file_size, None, None

        line = csv_file.readline()
        while line and not line.strip():
            row_offset = csv_file.tell()
            line = csv_file.readline()
        if not line:
            return file_size, None, None

        first_field = line.split(b',', 1)[0].strip().decode('utf-8', errors='ignore')
        return row_offset, line, normalize_index(first_field)

    def find_first_row_ge(csv_file, target, file_size, header_end):
        left = header_end
        right = file_size

        while left < right:
            mid = (left + right) // 2
            row_offset, line, row_index = read_row_at_or_after(csv_file, mid, file_size, header_end)
            if row_index is None:
                right = mid
                continue

            if row_index >= target:
                right = mid
            else:
                left = max(row_offset + len(line), mid + 1)

        row_offset, _, row_index = read_row_at_or_after(csv_file, left, file_size, header_end)
        if row_index is None or row_index < target:
            return file_size
        return row_offset

    def find_first_row_gt(csv_file, target, file_size, header_end):
        left = header_end
        right = file_size

        while left < right:
            mid = (left + right) // 2
            row_offset, line, row_index = read_row_at_or_after(csv_file, mid, file_size, header_end)
            if row_index is None:
                right = mid
                continue

            if row_index > target:
                right = mid
            else:
                left = max(row_offset + len(line), mid + 1)

        row_offset, _, row_index = read_row_at_or_after(csv_file, left, file_size, header_end)
        if row_index is None:
            return file_size
        if row_index > target:
            return row_offset
        return file_size

    def move_offset_backward_by_rows(csv_file, offset, rows, header_end):
        if rows <= 0 or offset <= header_end:
            return max(offset, header_end)

        search_end = offset
        remaining_newlines = rows + 1
        block_size = 4096

        while remaining_newlines > 0 and search_end > header_end:
            read_start = max(header_end, search_end - block_size)
            csv_file.seek(read_start)
            chunk = csv_file.read(search_end - read_start)
            relative_end = len(chunk)

            while remaining_newlines > 0:
                newline_pos = chunk.rfind(b'\n', 0, relative_end)
                if newline_pos < 0:
                    break
                relative_end = newline_pos
                remaining_newlines -= 1
                if remaining_newlines == 0:
                    return max(read_start + newline_pos + 1, header_end)

            if read_start == header_end:
                break

            search_end = read_start

        return header_end

    start_value = normalize_index(start_index)
    end_value = normalize_index(end_index)
    if start_value is not None and end_value is not None and start_value > end_value:
        raise ValueError("start_index must be less than or equal to end_index")

    with open(file_path, 'rb') as csv_file:
        header = csv_file.readline()
        if not header:
            raise ValueError(f"CSV file has no header: {file_path}")

        header_end = csv_file.tell()
        csv_file.seek(0, os.SEEK_END)
        file_size = csv_file.tell()

        start_offset = header_end if start_value is None else find_first_row_ge(csv_file, start_value, file_size, header_end)
        if start_offset >= file_size:
            return pd.read_csv(io.StringIO(header.decode('utf-8', errors='ignore')))

        if start_value is not None and prepend_n_rows > 0:
            start_offset = move_offset_backward_by_rows(csv_file, start_offset, prepend_n_rows, header_end)

        end_offset = file_size if end_value is None else find_first_row_gt(csv_file, end_value, file_size, header_end)
        if end_offset <= start_offset:
            return pd.read_csv(io.StringIO(header.decode('utf-8', errors='ignore')))

        csv_file.seek(start_offset)
        payload = csv_file.read(end_offset - start_offset)

    csv_text = (header + payload).decode('utf-8', errors='ignore')
    df = pd.read_csv(io.StringIO(csv_text))
    return df

def get_last_date_from_csv_tail(file_path, n_lines=5, default=None, output_format=None):
    lines = read_last_lines(file_path, n_lines=n_lines)
    if not lines:
        return default

    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        date_str = line.split(',', 1)[0].strip()
        if date_str == 'date':
            continue
        dt = pd.to_datetime(date_str, errors='coerce')
        if pd.notna(dt):
            if output_format:
                return dt.strftime(output_format)
            return dt

    return default


def get_first_last_line_from_csv(file_path, scan_tail_lines=200):
    """Return (df, msg) for first/last non-empty data rows in a CSV.

    - Success: (DataFrame, '')
    - Failure: (empty DataFrame, error_message)
    """
    try:
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            return pd.DataFrame(), f"CSV file not found or empty: {file_path}"

        first_data_line = None
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as csv_file:
            header = csv_file.readline().strip()
            if not header:
                return pd.DataFrame(), f"CSV file has no header: {file_path}"

            for line in csv_file:
                stripped = line.strip()
                if stripped:
                    first_data_line = stripped
                    break

        if first_data_line is None:
            return pd.read_csv(io.StringIO(header)), ''

        tail_lines = read_last_lines(file_path, n_lines=scan_tail_lines)
        last_data_line = None
        for line in reversed(tail_lines):
            stripped = line.strip()
            if stripped and stripped != header:
                last_data_line = stripped
                break

        if last_data_line is None:
            last_data_line = first_data_line

        data_lines = [first_data_line]
        if last_data_line != first_data_line:
            data_lines.append(last_data_line)

        csv_text = '\n'.join([header, *data_lines])
        return pd.read_csv(io.StringIO(csv_text)), ''
    except Exception as e:
        return pd.DataFrame(), f"get_first_last_line_from_csv failed: {e}"




def read_last_lines(file_path, n_lines=5):
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return []

    with open(file_path, 'rb') as f:
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        block_size = 4096
        data = b''
        while file_size > 0 and data.count(b'\n') <= n_lines:
            read_size = min(block_size, file_size)
            file_size -= read_size
            f.seek(file_size)
            data = f.read(read_size) + data

    lines = data.decode('utf-8', errors='ignore').splitlines()
    lines = [line for line in lines if line.strip()]
    return lines[-n_lines:]


def read_csv_by_tail(file_path, n_lines=5, start_index=None, end_index=None):
    lines = read_last_lines(file_path, n_lines=n_lines)
    if not lines:
        raise FileNotFoundError(f"CSV file not found or empty: {file_path}")

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as csv_file:
        header = csv_file.readline().strip()
    if not header:
        raise ValueError(f"CSV file has no header: {file_path}")

    csv_text = '\n'.join([header, *lines])
    df = pd.read_csv(io.StringIO(csv_text))

    if start_index is None and end_index is None:
        return df

    date_txt = df['date'].astype('string').fillna('').str.strip().str.slice(0, 10)
    mask = pd.Series(True, index=df.index)
    if start_index is not None:
        mask &= date_txt >= str(start_index)
    if end_index is not None:
        mask &= date_txt <= str(end_index)
    return df.loc[mask].reset_index(drop=True)


if __name__ == "__main__":
    import time
    from env_setting import ROOT
    file_path = f'{ROOT}/business_models/kline_models/kline_dataset_build/dataset/kline_uniform_samples/sh.603248.csv'

    st = time.time()
    last_lines = read_last_lines(file_path, n_lines=60)
    cost = time.time() - st
    print(f"Read last 60 lines: {len(last_lines)} lines, cost {cost:.8f} seconds")

    st = time.time()
    df = read_csv_by_index(file_path)
    cost = time.time() - st
    print(f"Read whole file: {len(df)} rows, cost {cost:.8f} seconds")

    st = time.time()
    df = read_csv_by_index(file_path, start_index='2026-03-05', end_index='2026-03-05', prepend_n_rows=60)
    cost = time.time() - st
    print(f"Read 6 months with 60 rows prepended: {len(df)} rows, cost {cost:.8f} seconds")
    print(df)