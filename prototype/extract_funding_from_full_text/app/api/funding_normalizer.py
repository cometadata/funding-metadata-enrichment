import re
import unicodedata
from typing import List, Tuple, Set


def normalize_funding_statement(statement: str) -> str:
    """
    Normalize a funding statement by correcting whitespace, accents, 
    markdown escaping, and removing line numbers.
    """
    # Extract line ending numbers
    line_ending_numbers = set()
    lines = statement.split('\n')
    for line in lines:
        match = re.search(r'\b(\d+)\s*$', line)
        if match:
            line_ending_numbers.add(int(match.group(1)))

    # Normalize whitespace
    normalized = re.sub(r'\s+', ' ', statement)

    # Fix markdown escaping
    normalized = normalized.replace(r'\_', '_')
    normalized = normalized.replace(r'\*', '*')
    normalized = normalized.replace(r'\[', '[')
    normalized = normalized.replace(r'\]', ']')

    # Fix accents
    accent_replacements = [
        (r'C´\s*aceres', 'Cáceres'),
        (r'C´aceres', 'Cáceres'),
        (r'([A-Za-z])´\s*([a-z])', r'\1\2'),
        (r'a`', 'à'),
        (r'e`', 'è'),
        (r'e´', 'é'),
        (r'a´', 'á'),
        (r'o´', 'ó'),
        (r'u´', 'ú'),
        (r'n~', 'ñ'),
        (r'N~', 'Ñ'),
    ]

    for pattern, replacement in accent_replacements:
        normalized = re.sub(pattern, replacement, normalized)

    normalized = unicodedata.normalize('NFC', normalized)

    # Remove sequential line numbers
    integers = re.findall(r'\b\d+\b', normalized)

    if len(integers) >= 3:
        int_positions = []
        for match in re.finditer(r'\b(\d+)\b', normalized):
            int_positions.append(
                (int(match.group(1)), match.start(), match.end()))

        sequences_to_remove = []
        i = 0
        while i < len(int_positions) - 2:
            num1, start1, end1 = int_positions[i]
            num2, start2, end2 = int_positions[i + 1]
            num3, start3, end3 = int_positions[i + 2]

            if num2 == num1 + 1 and num3 == num2 + 1:
                sequence_positions = [
                    (start1, end1), (start2, end2), (start3, end3)]
                sequence_numbers = [num1, num2, num3]
                j = i + 3
                while j < len(int_positions):
                    num_next, start_next, end_next = int_positions[j]
                    if num_next == int_positions[j-1][0] + 1:
                        sequence_positions.append((start_next, end_next))
                        sequence_numbers.append(num_next)
                        j += 1
                    else:
                        break

                is_line_number_sequence = False

                if all(num in line_ending_numbers for num in sequence_numbers):
                    is_line_number_sequence = True
                else:
                    all_look_like_line_numbers = True
                    for idx, (start, end) in enumerate(sequence_positions):
                        num = sequence_numbers[idx]
                        before_context = normalized[max(0, start-10):start]
                        after_context = normalized[end:min(
                            len(normalized), end+10)]

                        has_line_number_pattern = False

                        if re.search(r'and\s+$', before_context) and num in line_ending_numbers:
                            has_line_number_pattern = True
                        elif re.search(r'^\s*-\s*', after_context):
                            has_line_number_pattern = True
                        elif end == len(normalized):
                            has_line_number_pattern = True

                        if not has_line_number_pattern:
                            all_look_like_line_numbers = False
                            break

                    if all_look_like_line_numbers:
                        is_line_number_sequence = True

                if is_line_number_sequence:
                    sequences_to_remove.extend(sequence_positions)

                i = j
            else:
                i += 1

        # Remove identified sequences
        for start, end in reversed(sequences_to_remove):
            before_char = normalized[start-1] if start > 0 else ''
            after_char = normalized[end] if end < len(normalized) else ''

            remove_start = start
            remove_end = end

            while remove_start > 0 and normalized[remove_start-1] == ' ':
                remove_start -= 1
                if remove_start > 0 and normalized[remove_start-1].isalnum():
                    break

            if end < len(normalized) - 1 and normalized[end:end+2] in ['- ', ' -']:
                remove_end = end + 2

            if (remove_start > 0 and normalized[remove_start-1].isalnum() and
                    remove_end < len(normalized) and normalized[remove_end].isalnum()):
                normalized = normalized[:remove_start] + \
                    ' ' + normalized[remove_end:]
            else:
                normalized = normalized[:remove_start] + \
                    normalized[remove_end:]

    # Final whitespace cleanup
    normalized = re.sub(r'\s+', ' ', normalized)
    normalized = normalized.strip()

    return normalized


def is_improperly_formatted(statement: str) -> bool:
    """Check if a statement has problematic formatting (e.g., tables)."""
    # Check for table separators
    if '|-----|' in statement or '|---|' in statement:
        return True

    # Check for table cells with numbers
    if re.search(r'\|\s*\d+\s*\|', statement):
        return True

    # Check for excessive pipes (likely a table)
    pipe_count = statement.count('|')
    if pipe_count > 4:
        return True

    # Check for multiple lines with pipes
    lines = statement.split('\n')
    pipe_lines = [line for line in lines if '|' in line]
    if len(pipe_lines) > 3:
        return True

    return False


def extract_unique_funding_statements(
    funding_statements: List[dict], 
    normalize: bool = True,
    exclude_problematic: bool = False
) -> Tuple[List[str], List[str]]:
    """
    Extract unique funding statements from a list of funding records.
    Returns (clean_statements, problematic_statements).
    """
    clean_statements = set()
    problematic_statements = set()

    for funding in funding_statements:
        full_para = funding.get('full_paragraph', '').strip()
        if full_para:
            if is_improperly_formatted(full_para):
                problematic_statements.add(full_para)
            else:
                if normalize:
                    full_para = normalize_funding_statement(full_para)
                clean_statements.add(full_para)

    return sorted(list(clean_statements)), sorted(list(problematic_statements))