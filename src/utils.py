import subprocess
import sys


def get_dependencies():
    try:
        # Run pip freeze and capture its output
        result = subprocess.run(
            ["pip", "freeze"], capture_output=True, text=True, check=True
        )

        # Split the output into lines
        lines = result.stdout.split("\n")

        # Filter out comments and empty lines
        filtered_lines = [
            line for line in lines if not line.startswith("#") and line.strip()
        ]

        return "\n".join(filtered_lines)

    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running pip freeze: {e}")
        sys.exit(1)

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)
