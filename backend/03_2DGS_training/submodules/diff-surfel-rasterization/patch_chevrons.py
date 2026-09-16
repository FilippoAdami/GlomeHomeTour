import os
import re

for root, dirs, files in os.walk("."):
    for file in files:
        if file.endswith(".hip"):
            filepath = os.path.join(root, file)
            with open(filepath, "r") as f:
                content = f.read()

            # Deterministically find chevron pairs separated by ANY whitespace
            patched = re.sub(r'<<\s+<', '<<<', content)
            patched = re.sub(r'>>\s+>', '>>>', patched)

            if patched != content:
                with open(filepath, "w") as f:
                    f.write(patched)
                print(f"Successfully patched syntax in: {filepath}")