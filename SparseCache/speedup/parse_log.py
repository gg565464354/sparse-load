import re
import json
from pprint import pprint

def parse_log_file(log_file_path):
    """
    Parse log file with entries in the format:
    ###Log name=MyCache input_len=2000 output_len=10 bsz=1 total=1.069 prefill=0.373 decode=0.696
    """
    pattern = re.compile(
        r"###Log "
        r"name=(?P<name>\w+) "
        r"input_len=(?P<input_len>\d+) "
        r"output_len=(?P<output_len>\d+) "
        r"bsz=(?P<bsz>\d+) "
        r"total=(?P<total>[\d.]+) "
        r"prefill=(?P<prefill>[\d.]+) "
        r"decode=(?P<decode>[\d.]+)"
    )
    
    results = []
    
    with open(log_file_path, 'r') as f:
        for line in f:
            if line.startswith("###Log"):
                match = pattern.search(line)
                if match:
                    results.append({
                        'System': match.group('name'),
                        'Input': int(match.group('input_len')),
                        'Output': int(match.group('output_len')),
                        'Batch Size': int(match.group('bsz')),
                        'Total Time': float(match.group('total')),
                        'Prefill Time': float(match.group('prefill')),
                        'Decode Time': float(match.group('decode'))
                    })
    
    return results

def save_to_jsonl(data, output_file):
    """Save data to a JSONL file"""
    with open(output_file, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

# Example usage
log_file = "run_log.txt"
output_file = "results.jsonl"

parsed_results = parse_log_file(log_file)

# Print results
pprint(parsed_results)

parsed_results = parse_log_file(log_file)

# Save results to JSONL file
save_to_jsonl(parsed_results, output_file)
print(f"\nResults saved to {output_file}")


