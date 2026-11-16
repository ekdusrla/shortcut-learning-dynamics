"""Linux RSS sampling, isolated from unrelated experiment scripts."""
import os
from pathlib import Path


def tree_rss():
    """Sample Linux parent + descendant RSS; shared pages may be counted twice."""
    processes = {}
    for directory in Path('/proc').glob('[0-9]*'):
        try:
            fields = {}
            for line in (directory / 'status').read_text().splitlines():
                if line.startswith(('PPid:', 'VmRSS:')):
                    key, value = line.split(':', 1)
                    fields[key] = int(value.split()[0])
            processes[int(directory.name)] = (fields.get('PPid', -1), fields.get('VmRSS', 0))
        except (OSError, ValueError):
            continue
    selected = {os.getpid()}
    while True:
        expanded = selected | {pid for pid, (parent, _) in processes.items() if parent in selected}
        if expanded == selected:
            break
        selected = expanded
    return sum(processes.get(pid, (0, 0))[1] for pid in selected) * 1024
