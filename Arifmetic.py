import os
import time
from collections import defaultdict
import math

# Helper functions for varint encoding/decoding (same as before)
def write_varint(f, number):
    """Запись числа в формате varint."""
    while number > 0x7F:
        f.write(bytes([(number & 0x7F) | 0x80]))
        number >>= 7
    f.write(bytes([number]))

def read_varint(f):
    """Чтение числа в формате varint."""
    number = 0
    shift = 0
    while True:
        byte = f.read(1)
        if not byte:
            return None  # Indicate end of data
        byte = byte[0]
        number |= (byte & 0x7F) << shift
        shift += 7
        if not (byte & 0x80):
            break
    return number

def calculate_frequency(file_path):    
    """Вычисление частоты символов в файле."""
    frequency = defaultdict(int)
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            for char in text:
                frequency[char] += 1
    except FileNotFoundError:
        print(f"Ошибка: Файл не найден: {file_path}")
        return None
    except Exception as e:
        print(f"Ошибка при чтении файла: {e}")
        return None
    return dict(frequency)  # Ensure it's a regular dict

def encode_file(input_file, output_file):
    """Кодирование файла с использованием арифметического кодирования (целочисленная арифметика)."""
    start_time = time.time()
    print("--- Начало кодирования ---")    
    # 1. Calculate Frequencies
    frequency = calculate_frequency(input_file)
    if not frequency:
        print("Ошибка: Не удалось рассчитать частоты.")
        return

    total_symbols = sum(frequency.values())
    print(f"Общее количество символов: {total_symbols}")

    # 2. Prepare symbol order
    sorted_chars = sorted(frequency.keys())

    cumulative_prob_intervals = {}
    prob_table_for_print = []
    
    current_cumulative_prob = 0.0
    print("\nТаблица частот, вероятностей и накоп. вероятн. интервалов:")
    print("-"*80)
    print("{:<10} | {:<15} | {:<20} | {:<30}".format("Символ", "Частота", "Вероятность", "Накоп. интервал [low,high)"))
    print("-"*80)
    
    for char in sorted_chars:
        freq = frequency[char]
        probability = freq / total_symbols
        low_prob = current_cumulative_prob
        high_prob = current_cumulative_prob + probability
        
        cumulative_prob_intervals[char] = (low_prob, high_prob)
        prob_table_for_print.append({
            'char': char,
            'freq': freq,
            'prob': probability,
            'low_prob': low_prob,
            'high_prob': high_prob
        })
        print("{:<10} | {:<15} | {:<20.10f} | [{:<12.10f}, {:<12.10f})".format(
            repr(char), freq, probability, low_prob, high_prob
        ))
        current_cumulative_prob = high_prob

    # 3. Integer Arithmetic Coding
    range_bit_size = 32  # Adjust as needed
    full_range = 1 << range_bit_size
    quarter_range = full_range >> 2
    half_range = full_range >> 1
    low = 0
    high = full_range - 1

    pending_bits = 0  # Track pending bits for renormalization
    output_bytes = []

    def output_bit(bit):
        """Outputs a bit to the output byte stream."""
        nonlocal pending_bits, output_bytes
        output_bytes.append(bit)
        for _ in range(pending_bits):
            output_bytes.append(bit ^ 1)
        pending_bits = 0

    def output_pending_bits():
        """Outputs pending bits (for E3 scaling)."""
        nonlocal pending_bits
        pending_bits += 1    

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()

        # 4. Perform encoding
        for char in text:
            symbol = char
            symbol_low = 0
            symbol_high = 0
            cumulative_freq = 0

            # Find symbol range            
            for s in sorted_chars:
                if s == symbol:                    
                    symbol_low = cumulative_freq
                    symbol_high = cumulative_freq + frequency[s]
                    break
                cumulative_freq += frequency[s]

            symbol_range = symbol_high - symbol_low            
            new_range = high - low + 1

            high = low + (new_range * symbol_high) // total_symbols - 1
            low = low + (new_range * symbol_low) // total_symbols

            # Renormalization/Bit Stuffing
            while True:
                if high < half_range:
                    output_bit(0)
                    # Сброс pending-битов после вывода
                    for _ in range(pending_bits):
                        output_bit(1)
                    pending_bits = 0
                elif low >= half_range:
                    output_bit(1)
                    # Сброс pending-битов после вывода
                    for _ in range(pending_bits):
                        output_bit(0)
                    pending_bits = 0
                    low -= half_range
                    high -= half_range
                elif low >= quarter_range and high < 3 * quarter_range:
                    pending_bits += 1
                    low -= quarter_range
                    high -= quarter_range
                else:
                    break
                # Обновление диапазона
                low <<= 1
                high = (high << 1) | 1
                # high &= (full_range - 1)  # Ограничение размера

        # 5. Flush remaining bits
        pending_bits += 1
        if low < quarter_range:
            output_bit(0)
        else:
            output_bit(1)
        # Добавить дополнительный бит для завершения
        output_bit(1)
        output_bit(1)

        # 6. Write to output file        
        with open(output_file, 'wb') as f:
            # a) Write frequency table
            write_varint(f, len(sorted_chars))
            for char in sorted_chars:
                char_bytes = char.encode('utf-8')
                f.write(len(char_bytes).to_bytes(1, 'big'))                
                f.write(char_bytes)
                write_varint(f, frequency[char])

            # b) Write encoded data - pack bits into bytes
            packed_bytes = bytearray()
            current_byte = 0
            bit_count = 0
            for bit in output_bytes:                
                current_byte |= (bit << (7 - bit_count))
                bit_count += 1
                if bit_count == 8:
                    packed_bytes.append(current_byte)
                    current_byte = 0                    
                    bit_count = 0
            if bit_count > 0: # Pad the last byte if necessary
                packed_bytes.append(current_byte)

            f.write(bytes(packed_bytes))

    except FileNotFoundError:
        print("Ошибка: Входной файл не найден.")
        return
    except Exception as e:
        print(f"Ошибка во время кодирования: {e}")
        return

    end_time = time.time()    
    orig_size = os.path.getsize(input_file)
    comp_size = os.path.getsize(output_file)    
    print("\n--- Кодирование завершено ---")
    print(f"Исходный размер: {orig_size} байт")
    print(f"Сжатый размер: {comp_size} байт")
    print(f"Степень сжатия: {comp_size / orig_size:.3f}")
    print(f"Затраченное время: {end_time - start_time:.3f} сек")
    print("Символы в кодировщике:", sorted_chars)


def decode_file(encoded_file, output_file):
    """Исправленный декодер арифметического кода."""
    start_time = time.time()
    
    with open(encoded_file, 'rb') as f:
        # 1) Читаем модель
        num_symbols = read_varint(f)
        if num_symbols is None:
            raise ValueError("Не удалось прочитать число символов")
        
        frequency = {}
        sorted_chars = []
        for _ in range(num_symbols):
            # длина UTF-8 последовательности
            char_len = int.from_bytes(f.read(1), 'big')
            char = f.read(char_len).decode('utf-8')
            freq  = read_varint(f)
            frequency[char] = freq
            sorted_chars.append(char)
        sorted_chars.sort()
        total_symbols = sum(frequency.values())
        
        # 2) Настройка диапазонов
        RANGE_BITS   = 32
        full_range   = 1 << RANGE_BITS
        quarter      = full_range >> 2
        half         = full_range >> 1
        low, high    = 0, full_range - 1
        
        # 3) Читаем первые RANGE_BITS бит в value
        bit_buffer = []
        while len(bit_buffer) < RANGE_BITS:
            byte = f.read(1)
            if not byte:
                # дополняем нулями, если конец файла
                bit_buffer += [0] * (RANGE_BITS - len(bit_buffer))
                break
            b = byte[0]
            for i in range(7, -1, -1):
                bit_buffer.append((b >> i) & 1)
        value = 0
        for i in range(RANGE_BITS):
            value = (value << 1) | bit_buffer[i]
        bit_index = RANGE_BITS
        
        # 4) Основной цикл декодирования
        def next_bit():
            nonlocal bit_index, bit_buffer
            if bit_index >= len(bit_buffer):
                byte = f.read(1)
                if not byte:
                    bit_buffer.append(0)
                else:
                    b = byte[0]
                    for i in range(7, -1, -1):
                        bit_buffer.append((b >> i) & 1)
            bit = bit_buffer[bit_index]
            bit_index += 1
            return bit
        
        decoded = []
        for _ in range(total_symbols):
            # находим символ по scaled_value
            scaled = ((value - low) * total_symbols) // (high - low + 1)
            cum = 0
            for ch in sorted_chars:
                if cum <= scaled < cum + frequency[ch]:
                    symbol = ch
                    break
                cum += frequency[ch]
            decoded.append(symbol)
            
            # обновляем [low, high]
            sym_low  = cum
            sym_high = cum + frequency[symbol]
            rng      = high - low + 1
            high = low + (rng * sym_high)//total_symbols - 1
            low  = low + (rng * sym_low)//total_symbols
            
            # RENORMALIZATION (только один вызов next_bit)
            while True:
                if high < half:
                    # ничего не делаем с low/value
                    pass
                elif low >= half:
                    low  -= half
                    high -= half
                    value -= half
                elif low >= quarter and high < 3*quarter:
                    low  -= quarter
                    high -= quarter
                    value -= quarter
                else:
                    break
                low  <<= 1
                high = (high << 1) | 1
                # единожды читаем следующий бит
                value = (value << 1) | next_bit()
        
        # 5) Записываем результат
        with open(output_file, 'w', encoding='utf-8') as out:
            out.write(''.join(decoded))
    
    print("--- Декодирование завершено ---")
    print("Декодирование завершено за", time.time() - start_time, "сек")
    print("Символы в декодере:", sorted_chars)
    print("Частоты:", frequency)

def next_bit(f, bit_buffer, bit_index):
    """
    Возвращает (next_bit, new_bit_index), где:
      - next_bit: следующий бит (0 или 1)
      - new_bit_index: обновлённый индекс
    """
    # Если в буфере кончились биты — читаем очередной байт и расширяем буфер
    if bit_index >= len(bit_buffer):
        byte = f.read(1)
        if not byte:
            # В файле больше нет байт — считаем, что дальше идут нули
            return 0, bit_index
        b = byte[0]
        # Добавляем в буфер 8 новых бит (старший сначала)
        for i in range(7, -1, -1):
            bit_buffer.append((b >> i) & 1)
    # Отдаём следующий бит
    bit = bit_buffer[bit_index]
    return bit, bit_index + 1

def main():
    input_file = "input.txt"
    encoded_file = "encoded.bin"
    decoded_file = "decoded.txt"

    choice = input("Введите 1 - закодировать, 2 - декодировать: ")

    if choice == '1':
        encode_file(input_file, encoded_file)    
    elif choice == '2':
        decode_file(encoded_file, decoded_file)

    if choice == '2' and os.path.exists(input_file) and os.path.exists(decoded_file):
        try:
            with open(input_file, 'r', encoding='utf-8') as f1, open(decoded_file, 'r', encoding='utf-8') as f2:
                if f1.read() == f2.read():
                    print("файлы совпадают.")
                else:
                    print("файлы н совпадают после декодирования.")
        except Exception as e:
            print(f"Ошибка при сравнении файлов: {e}")
    else:
        print("Проверка равенства файлов пропущена.")


if __name__ == "__main__":
    main()