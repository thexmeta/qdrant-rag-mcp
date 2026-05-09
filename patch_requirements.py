with open('requirements.txt', 'r') as f:
    content = f.read()

content = content.replace("psutil>=5.9.0,<6.0.0  # System and process monitoring", "psutil>=5.9.0,<6.0.0  # System and process monitoring\nfastembed>=0.2.0,<1.0.0")

with open('requirements.txt', 'w') as f:
    f.write(content)
