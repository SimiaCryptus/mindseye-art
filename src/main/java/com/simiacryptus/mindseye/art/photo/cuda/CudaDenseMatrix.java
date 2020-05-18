/*
 * Copyright (c) 2019 by Andrew Charneski.
 *
 * The author licenses this file to you under the
 * Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance
 * with the License.  You may obtain a copy
 * of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.simiacryptus.mindseye.art.photo.cuda;

import com.simiacryptus.ref.lang.RefLazyVal;
import com.simiacryptus.ref.lang.ReferenceCountingBase;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcusolver.cusolverSpHandle;
import jcuda.jcusparse.cusparseHandle;
import jcuda.jcusparse.cusparseMatDescr;

import javax.annotation.Nonnull;

import static jcuda.jcusolver.JCusolverSp.cusolverSpCreate;
import static jcuda.jcusparse.JCusparse.*;
import static jcuda.jcusparse.cusparseIndexBase.CUSPARSE_INDEX_BASE_ZERO;
import static jcuda.runtime.JCuda.*;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice;

/**
 * The type Cuda dense matrix.
 */
public class CudaDenseMatrix extends RefLazyVal<CudaDenseMatrix.GpuCopy> {

  /**
   * The Matrix.
   */
  public final SparseMatrixFloat matrix;

  /**
   * Instantiates a new Cuda dense matrix.
   *
   * @param matrix the matrix
   */
  public CudaDenseMatrix(SparseMatrixFloat matrix) {
    this.matrix = matrix;
  }

  /**
   * New sparse handle cusparse handle.
   *
   * @return the cusparse handle
   */
  @Nonnull
  public static cusparseHandle newSparseHandle() {
    cusparseHandle handle = new cusparseHandle();
    cusparseCreate(handle);
    return handle;
  }

  /**
   * New solver handle cusolver sp handle.
   *
   * @return the cusolver sp handle
   */
  @Nonnull
  public static cusolverSpHandle newSolverHandle() {
    cusolverSpHandle handle = new cusolverSpHandle();
    cusolverSpCreate(handle);
    return handle;
  }

  /**
   * Descriptor cusparse mat descr.
   *
   * @param matType   the mat type
   * @param indexBase the index base
   * @return the cusparse mat descr
   */
  @Nonnull
  public static cusparseMatDescr descriptor(int matType, int indexBase) {
    cusparseMatDescr descra = new cusparseMatDescr();
    cusparseCreateMatDescr(descra);
    cusparseSetMatType(descra, matType);
    cusparseSetMatIndexBase(descra, indexBase);
    return descra;
  }

  /**
   * To device pointer.
   *
   * @param values the values
   * @return the pointer
   */
  @Nonnull
  public static Pointer toDevice(@Nonnull float[] values) {
    Pointer cooVal = new Pointer();
    cudaMalloc(cooVal, values.length * Sizeof.FLOAT);
    cudaMemcpy(cooVal, Pointer.to(values), values.length * Sizeof.FLOAT, cudaMemcpyHostToDevice);
    return cooVal;
  }

  /**
   * To device pointer.
   *
   * @param values the values
   * @return the pointer
   */
  @Nonnull
  public static Pointer toDevice(@Nonnull int[] values) {
    Pointer cooRowIndex = new Pointer();
    cudaMalloc(cooRowIndex, values.length * Sizeof.INT);
    cudaMemcpy(cooRowIndex, Pointer.to(values), values.length * Sizeof.INT, cudaMemcpyHostToDevice);
    return cooRowIndex;
  }

  @Override
  @Nonnull
  public CudaDenseMatrix.GpuCopy build() {
    return new GpuCopy(this);
  }

  public @SuppressWarnings("unused")
  void _free() {
    super._free();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  CudaDenseMatrix addRef() {
    return (CudaDenseMatrix) super.addRef();
  }

  /**
   * The type Gpu copy.
   */
  public static final class GpuCopy extends ReferenceCountingBase {
    /**
     * The Row indices.
     */
    @Nonnull
    public final Pointer rowIndices;
    /**
     * The Column indices.
     */
    @Nonnull
    public final Pointer columnIndices;
    /**
     * The Values.
     */
    @Nonnull
    public final Pointer values;
    /**
     * The Matrix.
     */
    public final SparseMatrixFloat matrix;
    /**
     * The Rows.
     */
    public final int rows;

    /**
     * Instantiates a new Gpu copy.
     *
     * @param cudaCoo the cuda coo
     */
    public GpuCopy(@Nonnull CudaDenseMatrix cudaCoo) {
      this.matrix = cudaCoo.matrix;
      rows = matrix.rows;
      rowIndices = toDevice(matrix.rowIndices);
      columnIndices = toDevice(matrix.colIndices);
      values = toDevice(matrix.values);
      cudaCoo.freeRef();
    }

    /**
     * Csr rows pointer.
     *
     * @param handle the handle
     * @return the pointer
     */
    @Nonnull
    public Pointer csrRows(cusparseHandle handle) {
      Pointer csrRowPtr = new Pointer();
      cudaMalloc(csrRowPtr, (rows + 1) * Sizeof.INT);
      cusparseXcoo2csr(handle, rowIndices, matrix.rowIndices.length, rows, csrRowPtr, CUSPARSE_INDEX_BASE_ZERO);
      return csrRowPtr;
    }

    public void _free() {
      super._free();
      cudaFree(this.rowIndices);
      cudaFree(this.columnIndices);
      cudaFree(this.values);
    }

    @Nonnull
    public @Override
    @SuppressWarnings("unused")
    GpuCopy addRef() {
      return (GpuCopy) super.addRef();
    }
  }
}
