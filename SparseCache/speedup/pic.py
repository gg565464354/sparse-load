import json
import matplotlib.pyplot as plt

# Read data from JSONL file
data = []
with open('results.jsonl', 'r') as f:
    for line in f:
        data.append(json.loads(line))

# Separate data by system
infini_gen = [d for d in data if d['System'] == 'InfiniGen']
my_cache = [d for d in data if d['System'] == 'MyCache']
# my_cache_cpu = [d for d in data if d['System'] == 'MyCache(cpu)']

# Extract data for plotting
batch_sizes = sorted({d['Batch Size'] for d in data})

def get_prefill_times(system_data):
    """Extract prefill times in order of batch sizes"""
    return [next(d['Prefill Time'] for d in system_data if d['Batch Size'] == bs) 
            for bs in batch_sizes]


def get_decode_times(system_data):
    """Extract decode times in order of batch sizes"""
    return [next(d['Decode Time'] for d in system_data if d['Batch Size'] == bs) 
            for bs in batch_sizes]

def get_total_times(system_data):
    """Extract decode times in order of batch sizes"""
    return [next(d['Decode Time'] for d in system_data if d['Batch Size'] == bs) 
            for bs in batch_sizes]

infini_cost = get_decode_times(infini_gen)
mycache_cost = get_decode_times(my_cache)
# mycache_cpu_cost = get_decode_times(my_cache_cpu)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(batch_sizes, infini_cost, marker='o', label='InfiniGen', color='blue')
plt.plot(batch_sizes, mycache_cost, marker='s', label='MyCache', color='orange')
# plt.plot(batch_sizes, mycache_cpu_cost, marker='s', label='MyCache(cpu)', color='green')

plt.title('Decode Time Comparison', fontsize=14)
plt.xlabel('Batch Size', fontsize=12)
plt.ylabel('Decode Time (seconds)', fontsize=12)
plt.xticks(batch_sizes)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.tight_layout()

# Save and show plot
plt.savefig('decode_time_comparison.png', dpi=300)
plt.show()