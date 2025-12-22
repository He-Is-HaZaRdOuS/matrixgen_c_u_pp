#!/usr/bin/env python3

import sys
import os

import matplotlib

matplotlib.use("Agg")  # Prevents GUI backend issues in some environments

import scipy.io as sio
import scipy.sparse as sp
import matplotlib.pyplot as plt

TEMP_VISUALIZATION_FOLDER = os.path.join(
    os.getcwd(), "static", "visualizations"
)
os.makedirs(TEMP_VISUALIZATION_FOLDER, exist_ok=True)

def visualize_matrix_spy(
        matrix, title="Matrix", filename="matrix_spy_plot.pdf"
):
    """
    Visualize a single matrix's sparsity pattern using matplotlib's spy plot and save it as a PDF or SVG.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.spy(matrix, markersize=1)
    ax.set_xlabel("Columns")
    ax.set_ylabel("Rows")
    plt.tight_layout()

    # Save as vector graphic (PDF or SVG)
    filepath = os.path.join(TEMP_VISUALIZATION_FOLDER, filename)
    plt.savefig(
        filepath, format=os.path.splitext(filepath)[1][1:]
    )  # infer format from extension
    plt.close(fig)
    print(f"[INFO] saved {filepath}")
    return filepath

def load_matrix(file_path):
    """Load a matrix from a .mtx file and ensure it's in CSR format with valid dtype."""
    print(f"[INFO] Loading matrix from {file_path}...")
    try:
        matrix = sio.mmread(file_path)

        if not sp.isspmatrix(matrix):
            matrix = sp.csr_matrix(matrix)
        else:
            matrix = matrix.tocsr()

        # Check if dtype is valid for sparse matrices
        if matrix.dtype.kind == "O":  # 'O' stands for object
            raise ValueError(
                f"Matrix has unsupported dtype=object (from file: {file_path})"
            )

        print(f"[INFO] {file_path} loaded successfully")
        return matrix
    except Exception as e:
        print(f"[ERROR] Error loading the matrix from {file_path}: {e}")
        return None

def describe_matrix(path, A):
    nrows, ncols = A.shape
    nnz = A.nnz
    density = nnz / (nrows * ncols)

    print(f"[INFO] matrix: {path}")
    print(f"  rows     : {nrows}")
    print(f"  cols     : {ncols}")
    print(f"  nnz      : {nnz}")
    print(f"  density  : {density:.6e}")
    print()


def main():
    if len(sys.argv) < 2:
        print("usage: visualize_matrix.py <matrix.mtx> [output_prefix]")
        sys.exit(1)

    matrix_path = sys.argv[1]
    prefix = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(
        os.path.basename(matrix_path)
    )[0]

    if not os.path.isfile(matrix_path):
        print(f"[ERROR] file not found: {matrix_path}")
        sys.exit(1)

    print("[INFO] loading matrix...")
    A = load_matrix(matrix_path)

    describe_matrix(matrix_path, A)

    print("[INFO] generating spy plot...")
    out_file = visualize_matrix_spy(
        A,
        title=prefix,
        filename=f"{prefix}",
    )

    print(f"[INFO] saved spy plot to: {out_file}")


if __name__ == "__main__":
    main()
