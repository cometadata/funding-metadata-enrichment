import re
import csv
import json
import argparse
import unicodedata
from pathlib import Path


def load_funding_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_dois(filepath):
    dois = set()
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dois.add(row['doi'])
    return dois


def filename_to_doi(filename):
    base = filename.replace('.md', '')

    if '_' in base and base.startswith('10_'):
        parts = base.split('_')
        if len(parts) >= 3:
            doi_prefix = f"{parts[0]}.{parts[1]}"
            doi_suffix = '.'.join(parts[2:])
            doi = f"{doi_prefix}/{doi_suffix}"
            return doi

    return base


def normalize_funding_statement(statement):
    line_ending_numbers = set()
    lines = statement.split('\n')
    for line in lines:
        match = re.search(r'\b(\d+)\s*$', line)
        if match:
            line_ending_numbers.add(int(match.group(1)))

    normalized = re.sub(r'\s+', ' ', statement)

    normalized = normalized.replace(r'\_', '_')
    normalized = normalized.replace(r'\*', '*')
    normalized = normalized.replace(r'\[', '[')
    normalized = normalized.replace(r'\]', ']')

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

    normalized = re.sub(r'\s+', ' ', normalized)
    normalized = normalized.strip()

    return normalized


def is_improperly_formatted(statement):
    if '|-----|' in statement or '|---|' in statement:
        return True

    if re.search(r'\|\s*\d+\s*\|', statement):
        return True

    pipe_count = statement.count('|')
    if pipe_count > 4:
        return True

    lines = statement.split('\n')
    pipe_lines = [line for line in lines if '|' in line]
    if len(pipe_lines) > 3:
        return True

    return False


def extract_unique_funding_statements(document, normalize=False):
    clean_statements = set()
    problematic_statements = set()

    for funding in document.get('funding_statements', []):
        full_para = funding.get('full_paragraph', '').strip()
        if full_para:
            if is_improperly_formatted(full_para):
                problematic_statements.add(full_para)
            else:
                if normalize:
                    full_para = normalize_funding_statement(
                        full_para)
                clean_statements.add(full_para)

    return sorted(list(clean_statements)), sorted(list(problematic_statements))


def convert_funding_to_simple_json(
    funding_data,
    dois_set=None,
    validate_dois=False,
    normalize=False,
    exclude_problematic=False
):
    clean_results = []
    problematic_results = []
    skipped_count = 0

    for doc in funding_data.get('results_by_document', []):
        filename = doc.get('filename', '')
        if not filename:
            continue

        clean_statements, problematic_statements = extract_unique_funding_statements(
            doc, normalize=normalize)

        if not clean_statements and not problematic_statements:
            continue

        doi = filename_to_doi(filename)

        if exclude_problematic and problematic_statements and not clean_statements:
            # Skip this document entirely if it only has problematic statements
            continue

        if clean_statements:
            result = {
                'doi': doi,
                'funding_statements': clean_statements
            }

            if validate_dois and dois_set:
                if doi not in dois_set:
                    print(f"Warning: DOI {doi} not found in dois.csv")
                    skipped_count += 1

            clean_results.append(result)
        elif not exclude_problematic and problematic_statements:
            # Include problematic statements in main output if not excluding
            result = {
                'doi': doi,
                'funding_statements': problematic_statements
            }
            clean_results.append(result)

        if problematic_statements:
            problematic_entry = {
                'doi': doi,
                'filename': filename,
                'funding_statements': problematic_statements,
                'file_path': doc.get('file_path', '')
            }
            problematic_results.append(problematic_entry)

    if skipped_count > 0:
        print(f"Note: {skipped_count} DOIs were not found in dois.csv")

    return clean_results, problematic_results


def main():
    parser = argparse.ArgumentParser(
        description='Convert funding extraction JSON to simple JSON format with DOIs and funding statements.'
    )
    parser.add_argument('-n', '--normalize', action='store_true',
                        help='Normalize funding statements to correct whitespace, accents, markdown escaping, and remove line numbers')
    parser.add_argument('-i', '--input', required=True,
                        help='Input funding JSON file')
    parser.add_argument(
        '-o', '--output', help='Primary output JSON file name (defaults to input basename + _converted.json)')
    parser.add_argument('-d', '--dois', required=True,
                        help='DOIs CSV file for validation')
    parser.add_argument('-e', '--exclude-problematic', action='store_true',
                        help='Exclude entries with problematic formatting from the output')

    args = parser.parse_args()

    funding_file = args.input
    dois_file = args.dois

    input_path = Path(funding_file)
    base_name = input_path.stem

    if args.output:
        output_file = args.output
    else:
        output_file = f"{base_name}_converted.json"

    problematic_file = f"{base_name}_problematic.json"

    if not Path(funding_file).exists():
        print(f"Error: {funding_file} not found")
        return

    print(f"Loading funding data from {funding_file}...")
    funding_data = load_funding_data(funding_file)

    dois_set = None
    if Path(dois_file).exists():
        print(f"Loading DOIs from {dois_file}...")
        dois_set = load_dois(dois_file)
        print(f"Loaded {len(dois_set)} DOIs for validation")
    else:
        print(f"Warning: {dois_file} not found, skipping DOI validation")

    if args.normalize:
        if args.exclude_problematic:
            print("Converting to simple JSON format with normalization and excluding problematic entries...")
        else:
            print("Converting to simple JSON format with normalization...")
    else:
        if args.exclude_problematic:
            print("Converting to simple JSON format and excluding problematic entries...")
        else:
            print("Converting to simple JSON format...")

    clean_results, problematic_results = convert_funding_to_simple_json(
        funding_data,
        dois_set=dois_set,
        validate_dois=True if dois_set else False,
        normalize=args.normalize,
        exclude_problematic=args.exclude_problematic
    )

    print(f"Saving {len(clean_results)} clean entries to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(clean_results, f, indent=2, ensure_ascii=False)

    if problematic_results:
        print(f"Saving {len(problematic_results)} problematic entries to {problematic_file}...")
        with open(problematic_file, 'w', encoding='utf-8') as f:
            json.dump(problematic_results, f, indent=2, ensure_ascii=False)

    print("\nConversion complete!")
    print(f"Total documents processed: {len(funding_data.get('results_by_document', []))}")
    print(f"Documents with clean funding statements: {len(clean_results)}")
    print(f"Documents with problematic formatting: {len(problematic_results)}")

    if clean_results:
        print("\nSample clean output:")
        sample = clean_results[0]
        print(json.dumps(sample, indent=2))

    if problematic_results:
        print("\nSample problematic output:")
        sample = problematic_results[0]
        display_sample = {
            'doi': sample['doi'],
            'filename': sample['filename'],
            'funding_statements': [s[:200] + '...' if len(s) > 200 else s for s in sample['funding_statements'][:1]]
        }
        print(json.dumps(display_sample, indent=2))


if __name__ == '__main__':
    main()
