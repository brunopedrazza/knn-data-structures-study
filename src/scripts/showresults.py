import sys

from helpers.utils import print_results


if __name__ == "__main__":
    try:
        file_name = sys.argv[1]
        print_results(file_name=file_name)
    except IndexError:
        print("\nPlease provide the file name to show results.")
